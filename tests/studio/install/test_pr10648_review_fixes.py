# SPDX-License-Identifier: AGPL-3.0-only
"""Review follow-ups to the prebuilt marker fast path.

Two defects the marker work exposed, each with the user situation it costs:

  * the llama marker rewriter asked os.chown for the owner as well as the group, which a
    non-root member of a group-shared install cannot grant. chown is all-or-nothing, so the
    group was not applied either and os.replace installed the member's primary group,
    leaving the marker unreadable to everyone else who shares the install. prebuilt_core
    was fixed for this; the llama writer was not, and this work gave it a new caller
    (whisper's slim pairing backfill).

  * UNSLOTH_PREBUILT_FULL_CHECK turns the no-network shortcut off for llama and whisper.
    Node never read it, so the one variable documented as "force a full revalidation" left
    a third of the runtime answered from its marker.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PACKAGE_ROOT / "studio" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LLAMA = _load("studio_install_llama_prebuilt_pr10648_fixes", "install_llama_prebuilt.py")
CORE = _load("studio_prebuilt_core_pr10648_fixes", "prebuilt_core.py")
NODE = _load("studio_install_node_prebuilt_pr10648_fixes", "install_node_prebuilt.py")

requires_chown = pytest.mark.skipif(not hasattr(os, "chown"), reason = "os.chown is POSIX only")


def _node_host():
    return NODE.HostInfo(
        system = "Linux",
        machine = "x64",
        node_os = "linux",
        node_arch = "x64",
        archive_ext = ".tar.gz",
        is_windows = False,
    )


def _node_tree(root: Path, host) -> None:
    node = NODE.node_binary_path(root, host)
    npm = NODE.npm_cli_path(root, host)
    node.parent.mkdir(parents = True, exist_ok = True)
    npm.parent.mkdir(parents = True, exist_ok = True)
    node.write_bytes(b"node" * 64)
    npm.write_bytes(b"npm" * 64)
    node.chmod(0o755)


@requires_chown
@pytest.mark.parametrize(
    "writer_name",
    ["llama._write_marker", "core.write_live_marker"],
)
def test_a_live_marker_rewrite_never_asks_for_the_owner(tmp_path, monkeypatch, writer_name):
    """Both marker rewriters must ask os.chown for the group alone.

    A shared install's marker is owned by whoever installed it. chown(2) refuses the whole
    call when an unprivileged caller names another owner, so asking for the owner loses the
    group too, and the next reader in that group cannot read the marker.
    """
    marker = tmp_path / "MARKER.json"
    marker.write_text('{"release_tag": "b10840"}\n', encoding = "utf-8")

    calls = []
    if writer_name == "llama._write_marker":
        module, write = LLAMA, lambda: LLAMA._write_marker(marker, {"release_tag": "b10841"})
    else:
        module, write = CORE, lambda: CORE.write_live_marker(marker, {"release_tag": "b10841"})

    monkeypatch.setattr(module.os, "chown", lambda path, uid, gid: calls.append((uid, gid)))
    write()

    assert calls, "the rewriter did not try to restore ownership at all"
    assert [uid for uid, _ in calls] == [-1] * len(calls), (
        f"{writer_name} asked for the owner ({calls}); a non-root member of a group-shared "
        "install cannot grant that, and chown then declines the group as well"
    )


@requires_chown
def test_the_group_survives_a_rewrite_the_owner_cannot_be_granted(tmp_path, monkeypatch):
    """The end state, not just the argument: a refused owner must not cost the group.

    os.chown is emulated with the kernel's own rule -- an unprivileged caller may not change
    the owner, and chown is all-or-nothing -- so this fails for any writer that names one.
    """
    marker = tmp_path / "MARKER.json"
    marker.write_text('{"release_tag": "b10840"}\n', encoding = "utf-8")
    shared_gid = marker.stat().st_gid
    applied = []

    def unprivileged_chown(path, uid, gid):
        if uid != -1:
            raise PermissionError(1, "Operation not permitted")
        applied.append(gid)

    monkeypatch.setattr(LLAMA.os, "chown", unprivileged_chown)
    assert LLAMA._write_marker(marker, {"release_tag": "b10841"}) is True
    assert applied == [shared_gid], "the group was never applied to the replacement"


def test_a_marker_rewrite_keeps_the_keys_the_about_tab_renders(tmp_path, monkeypatch):
    """release_tag and tag reach the user through /api/system/hardware and the About tab.

    A rewrite that drops them changes what Studio displays, so pin that they survive.
    """
    marker = tmp_path / "MARKER.json"
    marker.write_text('{"release_tag": "b10840", "tag": "b10840"}\n', encoding = "utf-8")
    monkeypatch.setattr(LLAMA.os, "chown", lambda path, uid, gid: None)

    assert LLAMA._write_marker(marker, {"release_tag": "b10841", "tag": "b10841"}) is True

    import json

    payload = json.loads(marker.read_text(encoding = "utf-8"))
    assert payload["release_tag"] == "b10841"
    assert payload["tag"] == "b10841"
    assert not list(tmp_path.glob("MARKER.json.tmp-*")), "a temp marker was left behind"


def test_node_honours_the_full_check_escape_hatch(tmp_path, monkeypatch):
    """UNSLOTH_PREBUILT_FULL_CHECK is the documented way to force a full revalidation.

    llama and whisper both turn their shortcut off for it. Node did not, so the variable
    silently covered two of the three runtimes.
    """
    host = _node_host()
    _node_tree(tmp_path, host)
    NODE.write_metadata(tmp_path, version = "24.17.0", asset = "x", sha256 = "y")
    NODE.record_runtime_verification(tmp_path, host, version = "24.17.0", npm_major = 11)

    meta = NODE.load_metadata(tmp_path)
    monkeypatch.delenv("UNSLOTH_PREBUILT_FULL_CHECK", raising = False)
    assert (
        NODE._recorded_runtime_matches(tmp_path, host, meta, "24.17.0") is True
    ), "the recorded runtime should answer when the bypass is unset"

    monkeypatch.setenv("UNSLOTH_PREBUILT_FULL_CHECK", "1")
    assert (
        NODE._recorded_runtime_matches(tmp_path, host, meta, "24.17.0") is False
    ), "UNSLOTH_PREBUILT_FULL_CHECK=1 must send Node down the spawning path"


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " on "])
def test_the_node_bypass_spells_it_the_way_llama_does(monkeypatch, value):
    """One variable, one spelling. A hatch that answers differently per component is a hatch
    nobody can rely on."""
    monkeypatch.setenv("UNSLOTH_PREBUILT_FULL_CHECK", value)
    assert NODE.prebuilt_full_check_requested() is True
    assert LLAMA.prebuilt_full_check_requested() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
def test_the_node_bypass_stays_off_for_anything_else(monkeypatch, value):
    """An unset or negative value must leave the fast path in place, or every update pays
    the spawns forever."""
    monkeypatch.setenv("UNSLOTH_PREBUILT_FULL_CHECK", value)
    assert NODE.prebuilt_full_check_requested() is False
    assert LLAMA.prebuilt_full_check_requested() is False


# The migration run: the marker predates the runtime record, so there is no digest to check
# and, on Windows, no loader preflight either. That run is what decides whether the bytes on
# disk become the record every later run is compared against.
_LOGIC = _load("studio_install_llama_prebuilt_pr10648_logic", "install_llama_prebuilt.py")


def _legacy_marker_install(tmp_path, helpers, windows: bool):
    """A healthy tree under a marker with no runtime_files, i.e. every install made before
    this record existed."""
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    if windows:
        helpers.write_windows_install_shape(install_dir, include_llama_dll = True)
        host = helpers.windows_host()
        choice = helpers.asset_choice(
            name = "llama-b9001-bin-win-cpu-x64.zip",
            url = "https://example.com/x.zip",
            source_label = "published",
            install_kind = "windows-cpu",
        )
        repo = helpers.PREBUILT
    else:
        helpers.write_linux_install_shape(install_dir)
        host = helpers.linux_host()
        choice = helpers.asset_choice()
        repo = helpers.UPSTREAM
    checksums = helpers.release_checksums((choice.name, choice.expected_sha256, repo))
    plan = helpers.release_plan([choice], checksums)
    helpers.write_metadata(install_dir, choice, checksums)
    # Strip the keys that did not exist before this work, which is what every marker on disk
    # from an older Studio actually looks like. runtime_files is not a fingerprint input, so
    # removing it leaves the marker self-consistent -- exactly the legacy shape.
    import json as _json

    marker_path = install_dir / "UNSLOTH_PREBUILT_INFO.json"
    payload = _json.loads(marker_path.read_text(encoding = "utf-8"))
    payload.pop("runtime_files", None)
    payload.pop("host_profile", None)
    marker_path.write_text(_json.dumps(payload, indent = 2) + "\n", encoding = "utf-8")
    assert "runtime_files" not in (LLAMA.load_prebuilt_metadata(install_dir) or {})
    return install_dir, host, plan


@pytest.fixture(scope = "module")
def helpers():
    return _load_logic_helpers()


def _load_logic_helpers():
    import importlib.util as _iu

    path = Path(__file__).with_name("test_install_llama_prebuilt_logic.py")
    spec = _iu.spec_from_file_location("pr10648_logic_helpers", path)
    assert spec is not None and spec.loader is not None
    module = _iu.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_legacy_marker_is_not_blessed_without_asking_the_loader(tmp_path, monkeypatch, helpers):
    """A Windows install damaged under a pre-record marker must not have its damage recorded.

    Windows gets neither of the loader preflights below, and a marker with no runtime_files has
    no digest to fail, so before this the only integrity test was "the file is not zero bytes".
    The next run would then be compared against the damaged bytes for ever.
    """
    install_dir, host, plan = _legacy_marker_install(tmp_path, helpers, windows = True)

    probed = []
    monkeypatch.setattr(
        helpers.INSTALL_LLAMA_PREBUILT,
        "_binary_image_runs",
        lambda path, d, h, line = None: (probed.append(Path(path).name), False)[1],
    )
    assert (
        helpers.existing_install_matches_plan(install_dir, host, plan) is False
    ), "an image the OS refuses to start was accepted as a current install"
    assert probed, "the loader was never asked, so nothing checked the bytes"


def test_a_healthy_legacy_windows_install_still_migrates(tmp_path, monkeypatch, helpers):
    """The other half: the guard must not cost a working install its one migration run."""
    install_dir, host, plan = _legacy_marker_install(tmp_path, helpers, windows = True)

    probed = []
    monkeypatch.setattr(
        helpers.INSTALL_LLAMA_PREBUILT,
        "_binary_image_runs",
        lambda path, d, h, line = None: (probed.append(Path(path).name), True)[1],
    )
    assert helpers.existing_install_matches_plan(install_dir, host, plan) is True
    assert probed, "the migration run should have asked the loader once"


def test_a_linux_legacy_install_is_covered_by_its_preflight_instead(tmp_path, monkeypatch, helpers):
    """Linux and macOS already run a real preflight below, so they must not pay a spawn too."""
    install_dir, host, plan = _legacy_marker_install(tmp_path, helpers, windows = False)

    probed = []
    monkeypatch.setattr(
        helpers.INSTALL_LLAMA_PREBUILT,
        "_binary_image_runs",
        lambda path, d, h, line = None: (probed.append(Path(path).name), True)[1],
    )
    helpers.existing_install_matches_plan(install_dir, host, plan)
    assert probed == [], "Linux paid a --version spawn its ldd preflight already covers"

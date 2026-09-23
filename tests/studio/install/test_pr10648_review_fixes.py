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

import sys
from pathlib import Path

import pytest

from _pr10648_helpers import NEEDS_CHOWN as requires_chown
from _pr10648_helpers import load_studio_module as _load

LLAMA = _load("studio_install_llama_prebuilt_pr10648_fixes", "install_llama_prebuilt.py")
CORE = _load("studio_prebuilt_core_pr10648_fixes", "prebuilt_core.py")
NODE = _load("studio_install_node_prebuilt_pr10648_fixes", "install_node_prebuilt.py")


def _node_host():
    return NODE.HostInfo(
        system="Linux",
        machine="x64",
        node_os="linux",
        node_arch="x64",
        archive_ext=".tar.gz",
        is_windows=False,
    )


def _node_tree(root: Path, host) -> None:
    node = NODE.node_binary_path(root, host)
    npm = NODE.npm_cli_path(root, host)
    node.parent.mkdir(parents=True, exist_ok=True)
    npm.parent.mkdir(parents=True, exist_ok=True)
    node.write_bytes(b"node" * 64)
    npm.write_bytes(b"npm" * 64)
    node.chmod(0o755)


@requires_chown
@pytest.mark.parametrize(
    "writer_name",
    ["llama._write_marker", "core.write_live_marker"],
)
def test_a_live_marker_rewrite_falls_back_to_the_group_when_the_owner_is_refused(
    tmp_path, monkeypatch, writer_name
):
    """Both marker rewriters must ask for owner AND group, then for the group alone.

    Neither half is sufficient. A shared install's marker is owned by whoever installed it,
    and chown(2) refuses the WHOLE call when an unprivileged caller names another owner, so
    asking only for the owner loses the group too and the next reader in that group cannot
    read the marker. Asking only for the group is wrong the other way: under root, the one
    caller that CAN restore the owner, it leaves the marker owned by root, and an 0600
    marker stops being readable by the user who owns the install.
    """
    marker = tmp_path / "MARKER.json"
    marker.write_text('{"release_tag": "b10840"}\n', encoding="utf-8")
    original = marker.stat()

    calls = []
    if writer_name == "llama._write_marker":
        module, write = LLAMA, lambda: LLAMA._write_marker(marker, {"release_tag": "b10841"})
    else:
        module, write = CORE, lambda: CORE.write_live_marker(marker, {"release_tag": "b10841"})

    def refusing(path, uid, gid):
        calls.append((uid, gid))
        if uid != -1:
            raise PermissionError("a non-root member may not give a file away")

    monkeypatch.setattr(module.os, "chown", refusing)
    write()

    assert calls == [(original.st_uid, original.st_gid), (-1, original.st_gid)], (
        f"{writer_name} asked for {calls}; it must try the owner first and fall back to the "
        "group, so a non-root member keeps the group and root restores the owner"
    )


@requires_chown
def test_the_group_survives_a_rewrite_the_owner_cannot_be_granted(tmp_path, monkeypatch):
    """The end state, not just the argument: a refused owner must not cost the group.

    os.chown is emulated with the kernel's own rule -- an unprivileged caller may not change
    the owner, and chown is all-or-nothing -- so this fails for any writer that names one.
    """
    marker = tmp_path / "MARKER.json"
    marker.write_text('{"release_tag": "b10840"}\n', encoding="utf-8")
    shared_gid = marker.stat().st_gid
    applied = []

    def unprivileged_chown(path, uid, gid):
        if uid != -1:
            raise PermissionError(1, "Operation not permitted")
        applied.append(gid)

    monkeypatch.setattr(LLAMA.os, "chown", unprivileged_chown)
    assert LLAMA._write_marker(marker, {"release_tag": "b10841"}) is True
    assert applied == [shared_gid], "the group was never applied to the replacement"


@requires_chown
def test_a_marker_rewrite_keeps_the_keys_the_about_tab_renders(tmp_path, monkeypatch):
    """release_tag and tag reach the user through /api/system/hardware and the About tab.

    A rewrite that drops them changes what Studio displays, so pin that they survive.
    """
    marker = tmp_path / "MARKER.json"
    marker.write_text('{"release_tag": "b10840", "tag": "b10840"}\n', encoding="utf-8")
    monkeypatch.setattr(LLAMA.os, "chown", lambda path, uid, gid: None)

    assert LLAMA._write_marker(marker, {"release_tag": "b10841", "tag": "b10841"}) is True

    import json

    payload = json.loads(marker.read_text(encoding="utf-8"))
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
    NODE.write_metadata(tmp_path, version="24.17.0", asset="x", sha256="y")
    NODE.record_runtime_verification(tmp_path, host, version="24.17.0", npm_major=11)

    meta = NODE.load_metadata(tmp_path)
    monkeypatch.delenv("UNSLOTH_PREBUILT_FULL_CHECK", raising=False)
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


# The migration run: a pre-record marker has no digest and, on Windows, no loader preflight, so
# this run decides whether the bytes on disk become the reference every later run is held to.
_LOGIC = _load("studio_install_llama_prebuilt_pr10648_logic", "install_llama_prebuilt.py")


def _legacy_marker_install(tmp_path, helpers, windows: bool):
    """A healthy tree under a marker with no runtime_files, i.e. every install made before
    this record existed."""
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir()
    if windows:
        helpers.write_windows_install_shape(install_dir, include_llama_dll=True)
        host = helpers.windows_host()
        choice = helpers.asset_choice(
            name="llama-b9001-bin-win-cpu-x64.zip",
            url="https://example.com/x.zip",
            source_label="published",
            install_kind="windows-cpu",
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
    # Strip the keys that postdate this work, which is what an older Studio's marker looks like.
    # runtime_files is not a fingerprint input, so the stripped marker stays self-consistent.
    import json as _json

    marker_path = install_dir / "UNSLOTH_PREBUILT_INFO.json"
    payload = _json.loads(marker_path.read_text(encoding="utf-8"))
    payload.pop("runtime_files", None)
    payload.pop("host_profile", None)
    marker_path.write_text(_json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    assert "runtime_files" not in (LLAMA.load_prebuilt_metadata(install_dir) or {})
    return install_dir, host, plan


@pytest.fixture(scope="module")
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
    install_dir, host, plan = _legacy_marker_install(tmp_path, helpers, windows=True)

    probed = []
    monkeypatch.setattr(
        helpers.INSTALL_LLAMA_PREBUILT,
        "_binary_image_runs",
        lambda path, d, h, line=None: (probed.append(Path(path).name), False)[1],
    )
    assert (
        helpers.existing_install_matches_plan(install_dir, host, plan) is False
    ), "an image the OS refuses to start was accepted as a current install"
    assert probed, "the loader was never asked, so nothing checked the bytes"


def test_a_healthy_legacy_windows_install_still_migrates(tmp_path, monkeypatch, helpers):
    """The other half: the guard must not cost a working install its one migration run."""
    install_dir, host, plan = _legacy_marker_install(tmp_path, helpers, windows=True)

    probed = []
    monkeypatch.setattr(
        helpers.INSTALL_LLAMA_PREBUILT,
        "_binary_image_runs",
        lambda path, d, h, line=None: (probed.append(Path(path).name), True)[1],
    )
    assert helpers.existing_install_matches_plan(install_dir, host, plan) is True
    assert probed, "the migration run should have asked the loader once"


def test_a_linux_legacy_install_is_covered_by_its_preflight_instead(tmp_path, monkeypatch, helpers):
    """Linux and macOS already run a real preflight below, so they must not pay a spawn too."""
    install_dir, host, plan = _legacy_marker_install(tmp_path, helpers, windows=False)

    probed = []
    monkeypatch.setattr(
        helpers.INSTALL_LLAMA_PREBUILT,
        "_binary_image_runs",
        lambda path, d, h, line=None: (probed.append(Path(path).name), True)[1],
    )
    helpers.existing_install_matches_plan(install_dir, host, plan)
    assert probed == [], "Linux paid a --version spawn its ldd preflight already covers"


def test_a_marker_that_vanishes_between_the_two_reads_does_not_crash(tmp_path, monkeypatch):
    """_backfill_fingerprint_inputs reads the marker a second time.

    Another installer swapping the tree in between leaves None, and the old code called
    .get on it. The backfill has nothing to catch up at that point, so falling out is the
    answer -- the run that replaced the tree wrote its own marker.
    """
    calls = []

    class Ops:
        def load_prebuilt_metadata(self, install_dir):
            # Present for _kept_marker_patch's read, gone for the backfill's.
            calls.append(1)
            return {"install_fingerprint": "abc"} if len(calls) == 1 else None

        def metadata_path(self, install_dir):
            return tmp_path / "MARKER.json"

    selection = type(
        "Sel",
        (),
        {
            "coverage": {"min_os": None},
            "walk_back": None,
            "platform_os": "linux",
            "platform_arch": "x64",
            "fingerprint": lambda self: "abc",
        },
    )()

    monkeypatch.setattr(CORE, "_kept_marker_patch", lambda ops, d, s: {"fingerprint_coverage": {}})
    # The bug was an AttributeError here, not a wrong answer.
    CORE._backfill_fingerprint_inputs(Ops(), tmp_path, selection)


def test_a_blocked_marker_swap_is_retried_then_reported(tmp_path, monkeypatch):
    """Rename-over needs DELETE access on the destination, so a scanner holding the marker
    open fails the swap where the in-place write this replaced would have succeeded."""
    source = tmp_path / "tmp-marker"
    source.write_text("{}", encoding="utf-8")
    destination = tmp_path / "MARKER.json"

    blocked = OSError(13, "in use")
    blocked.winerror = 32
    attempts = []

    def blocked_then_ok(src, dst):
        attempts.append(1)
        if len(attempts) < 3:
            raise blocked
        Path(src).rename(dst)

    # Swap the MODULE's os reference, never os.name itself: pathlib reads os.name at runtime, so
    # setting it globally makes every path in the process a backslash one.
    class FakeOs:
        name = "nt"
        replace = staticmethod(blocked_then_ok)

        def __getattr__(self, item):
            return getattr(os, item)

    monkeypatch.setattr(CORE, "os", FakeOs())
    monkeypatch.setattr(CORE.time, "sleep", lambda _s: None)

    CORE.atomic_replace_from_tempfile(source, destination)
    assert len(attempts) == 3, "the blocked swap was not retried"
    assert destination.is_file()


def test_a_swap_blocked_for_any_other_reason_raises_at_once(tmp_path, monkeypatch):
    """A real problem must not be turned into an eight-step stall."""
    source = tmp_path / "tmp-marker"
    source.write_text("{}", encoding="utf-8")
    attempts = []

    def replace(src, dst):
        attempts.append(1)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(CORE.os, "replace", replace)
    with pytest.raises(OSError):
        CORE.atomic_replace_from_tempfile(source, tmp_path / "MARKER.json")
    assert len(attempts) == 1, "a non-transient failure was retried"


# The walk-back records why AUTOMATIC selection took an older release, so reusing it for a run
# that NAMED one answers "already matches N" while N is not installed. The full path turns off
# older-release fallback as soon as published_release_tag is supplied.
def _mac(macos_version=(14, 7)):
    return LLAMA.HostInfo(
        system="Darwin",
        machine="arm64",
        is_linux=False,
        is_windows=False,
        is_macos=True,
        is_x86_64=False,
        is_arm64=True,
        nvidia_smi=None,
        driver_cuda_version=None,
        compute_caps=[],
        visible_cuda_devices=None,
        has_physical_nvidia=False,
        has_usable_nvidia=False,
        macos_version=macos_version,
    )


def _walked_back_marker(
    host,
    *,
    installed="b9998",
    skipped="b9999",
):
    return {
        "release_tag": installed,
        "tag": installed,
        "walked_back_from": skipped,
        "walked_back_on_macos": CORE.macos_version_label(host),
    }


def test_a_pinned_release_is_not_satisfied_by_a_walked_back_install():
    host = _mac()
    marker = _walked_back_marker(host)
    assert LLAMA._release_expectation_met(marker, "b9999", host, pinned=True) is False


def test_an_automatic_request_still_accepts_the_walk_back():
    """The optimisation this PR exists for must survive the fix above: without a pin, a Mac
    holding the older release because the newest needs a newer OS is still current."""
    host = _mac()
    marker = _walked_back_marker(host)
    assert LLAMA._release_expectation_met(marker, "b9999", host, pinned=False) is True


def test_a_pinned_release_that_is_actually_installed_is_still_current():
    """The fix must not send every pinned request down the full path."""
    host = _mac()
    marker = _walked_back_marker(host, installed="b9999", skipped="b9999")
    assert LLAMA._release_expectation_met(marker, "b9999", host, pinned=True) is True


def test_the_pin_reaches_the_expectation_check(tmp_path, monkeypatch):
    """The guard is only worth anything if the caller actually passes the pin, so drive the
    real entry point and record what it asked."""
    seen = []
    monkeypatch.setattr(
        LLAMA,
        "_release_expectation_met",
        lambda marker, expected, host, *, pinned=False: seen.append(pinned) or False,
    )
    host = _mac()
    # Everything the guards AHEAD of the release check demand: failing an earlier one would
    # make this pass vacuously, with the pin never computed at all.
    monkeypatch.setattr(
        LLAMA,
        "load_prebuilt_metadata",
        lambda *_a, **_k: {
            "install_fingerprint": "x" * 64,
            "backend_request": "auto",
            "force_cpu": False,
            "prebuilt_fallback_used": False,
            "published_repo": LLAMA.DEFAULT_PUBLISHED_REPO,
            "backend": "metal",
            "host_profile": LLAMA.host_profile(host),
        },
    )
    monkeypatch.setattr(LLAMA, "_marker_backend_fits_host", lambda *_a, **_k: True)
    monkeypatch.setattr(LLAMA, "_runtime_preference_moved", lambda *_a, **_k: False)
    route = LLAMA.BackendRoute(
        backend="auto",
        host=host,
        published_repo=LLAMA.DEFAULT_PUBLISHED_REPO,
        published_release_tag="b9999",
        persist_llama_backend=None,
        persist_rocm_gfx=None,
    )
    LLAMA.existing_install_current_without_plan(
        tmp_path,
        llama_tag="latest",
        published_repo=LLAMA.DEFAULT_PUBLISHED_REPO,
        published_release_tag="b9999",
        backend_request="auto",
        force_cpu=False,
        route=route,
    )
    assert seen == [True], f"the pin never reached the expectation check: {seen}"


# The mode restore protects a REFRESH, which has another reader. A first write has neither a
# mode to preserve nor a reader, and raising there aborts a whole Node install over a cosmetic
# call, reachable on Windows through the sharing violation the swap already retries.
def _refuse_mode_change(monkeypatch, module):
    def refuse(path, mode):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(module.os, "chmod", refuse)


def test_a_first_node_marker_survives_a_refused_mode_restore(tmp_path, monkeypatch):
    install_dir = tmp_path / "node"
    install_dir.mkdir()
    _refuse_mode_change(monkeypatch, NODE)
    NODE.write_metadata(install_dir, version="24.17.0", asset="a.tar.gz", sha256="b" * 64)
    assert NODE.load_metadata(install_dir)["version"] == "24.17.0"
    assert not list(install_dir.glob("*.tmp-*")), "a temp marker was left behind"


def test_a_node_marker_refresh_abandons_on_a_refused_mode_restore(tmp_path, monkeypatch):
    """The refresh is where the harm is: the existing marker must be left exactly as it was,
    rather than replaced by one only its writer can read."""
    install_dir = tmp_path / "node"
    install_dir.mkdir()
    NODE.write_metadata(install_dir, version="24.17.0", asset="a.tar.gz", sha256="b" * 64)
    marker = NODE.metadata_path(install_dir)
    before = marker.read_bytes()

    _refuse_mode_change(monkeypatch, NODE)
    with pytest.raises(OSError):
        NODE.write_metadata(install_dir, version="25.0.0", asset="c.tar.gz", sha256="d" * 64)
    assert marker.read_bytes() == before, "the good marker was replaced anyway"
    assert not list(install_dir.glob("*.tmp-*")), "a temp marker was left behind"


def test_the_runtime_record_refresh_still_never_raises(tmp_path, monkeypatch):
    """record_runtime_verification is best effort by contract: an unrefreshable marker costs
    the two spawns again, and must never take the caller down with it."""
    host = _node_host()
    _node_tree(tmp_path, host)
    NODE.write_metadata(tmp_path, version="24.17.0", asset="a", sha256="b" * 64)
    _refuse_mode_change(monkeypatch, NODE)
    NODE.record_runtime_verification(tmp_path, host, version="24.17.0", npm_major=11)


# A release found untrustworthy must not read as "update unavailable"
# The keep paths exist because a lookup that could not ANSWER says nothing about the tree on disk.
# An asset outside the checksum index, or a manifest digest disagreeing with it, is the opposite:
# the release was fetched and found untrustworthy. Reporting "update unavailable, existing prebuilt
# kept" over that turns a tamper signal into a routine offline notice.
def _ops():
    """The one name expected_sha256_for reads, over prebuilt_core's own defaults. A component
    module supplies SHA256_ASSET_NAME; neither core nor llama defines it, and without it the
    call fails for a reason that has nothing to do with integrity."""
    return CORE.ModuleOps({**vars(CORE), "SHA256_ASSET_NAME": "whisper-prebuilt-sha256.json"})


def test_an_unverifiable_asset_raises_the_integrity_type():
    with pytest.raises(CORE.ReleaseIntegrityError):
        CORE.expected_sha256_for(_ops(), {}, "whisper-x.tar.gz")


def test_a_manifest_digest_disagreeing_with_the_index_raises_the_integrity_type():
    with pytest.raises(CORE.ReleaseIntegrityError):
        CORE.expected_sha256_for(
            _ops(),
            {"whisper-x.tar.gz": "a" * 64},
            "whisper-x.tar.gz",
            manifest_sha256="b" * 64,
        )


def test_the_integrity_type_is_still_a_prebuilt_fallback():
    """A subclass, so every existing `except PrebuiltFallback` keeps catching it and only the
    keep paths single it out. A sibling type would silently escape those handlers."""
    assert issubclass(CORE.ReleaseIntegrityError, CORE.PrebuiltFallback)


def test_a_plain_lookup_failure_is_not_an_integrity_error():
    """The distinction has to cut both ways, or the keep path stops working at all."""
    assert not isinstance(CORE.PrebuiltFallback("offline"), CORE.ReleaseIntegrityError)


# llama's keep arm must refuse an untrustworthy release too
# llama has kept an install on a failed lookup since before this branch, and that arm caught every
# PrebuiltFallback including the integrity refusals. Closed here so the two installers cannot
# disagree about what a keep is allowed to hide. The condition is inline, so this drives the real
# install_prebuilt: with the guard removed the whole install suite stayed green.
def _llama_keep_probe(monkeypatch, tmp_path, raised):
    """Run llama's install_prebuilt with the planner raising *raised*, over a tree its own
    _existing_install_runs accepts, and report whether it took the keep arm."""
    install_dir = tmp_path / "llama.cpp"
    (install_dir / "build" / "bin").mkdir(parents=True)
    for name in ("llama-server", "llama-quantize"):
        for target in (install_dir / name, install_dir / "build" / "bin" / name):
            target.write_bytes(b"#!/bin/sh\nexit 0\n")
            target.chmod(0o755)

    monkeypatch.setattr(LLAMA, "_existing_install_runs", lambda *_a, **_k: True)
    monkeypatch.setattr(
        LLAMA,
        "resolve_simple_install_release_plans",
        lambda *_a, **_k: (_ for _ in ()).throw(raised),
    )
    monkeypatch.setattr(LLAMA, "collect_system_report", lambda *_a, **_k: "")

    logs: list[str] = []
    monkeypatch.setattr(LLAMA, "log", lambda msg, *a, **k: logs.append(str(msg)))
    monkeypatch.setattr(LLAMA, "log_lines", lambda msg, *a, **k: logs.append(str(msg)))
    outcome = "returned"
    try:
        LLAMA.install_prebuilt(
            install_dir=install_dir,
            llama_tag="latest",
            published_repo=LLAMA.DEFAULT_PUBLISHED_REPO,
            published_release_tag="",
        )
    except SystemExit as exc:
        outcome = f"SystemExit({exc.code})"
    except BaseException as exc:
        outcome = f"{type(exc).__name__}"
    text = "\n".join(logs)
    return outcome, ("keeping the existing complete install" in text), text


def test_llama_keeps_the_install_when_the_lookup_could_not_answer(tmp_path, monkeypatch):
    outcome, kept, text = _llama_keep_probe(
        monkeypatch,
        tmp_path,
        LLAMA.PrebuiltFallback("network is unreachable"),
    )
    assert kept, f"an unavailable lookup should keep the install; got {outcome}\n{text[:400]}"


def test_llama_refuses_to_keep_over_an_untrustworthy_release(tmp_path, monkeypatch):
    outcome, kept, text = _llama_keep_probe(
        monkeypatch,
        tmp_path,
        LLAMA._core.ReleaseIntegrityError(
            "manifest sha256 for app-x.tar.gz disagrees with llama-prebuilt-sha256.json; "
            "refusing a possibly tampered release"
        ),
    )
    assert not kept, f"the keep arm swallowed an integrity failure; got {outcome}\n{text[:400]}"
    assert "tampered" in text, f"the reason never reached the user\n{text[:400]}"

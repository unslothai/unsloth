# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth studio update` must reuse the cache the install filled, not uv's default.

The installers set UV_CACHE_DIR (#10204) and storage_roots._setup_cache_env sets it
for the server; an update ran setup.sh/setup.ps1 from the CLI process and reached
neither, so it re-downloaded into uv's own default what the install had just fetched.

It must not overcorrect either: a shared-mode install leaves the wheels in uv's own
cache, and pointing the update at the empty Studio one costs a download online and
fails outright when uv may read only what is cached.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _studio():
    from unsloth_cli.commands import studio as _studio_mod
    return _studio_mod


def _setup_tree(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    (repo_root / "studio").mkdir(parents = True, exist_ok = True)
    (repo_root / "studio" / "setup.sh").write_text("")
    (repo_root / "studio" / "setup.ps1").write_text("")
    return repo_root


def _fill(
    cache_dir: Path,
    *,
    bucket: str = "archive-v0",
    name: str = "payload.so",
) -> Path:
    """Give a cache the shape uv gives it: a bucket directory holding package bytes."""
    leaf = cache_dir / bucket / "pkg"
    leaf.mkdir(parents = True, exist_ok = True)
    (leaf / name).write_bytes(b"\0" * 16)
    return cache_dir


@pytest.fixture
def caches(monkeypatch, tmp_path):
    """Both caches, cold, and uv's default answered without spawning uv.

    Every test states the state it needs; leaving the real machine's cache in play would
    make the outcome depend on whoever ran the suite.
    """
    studio = _studio()
    monkeypatch.delenv("UV_CACHE_DIR", raising = False)
    studio_home = tmp_path / "StudioHome"
    default_cache = tmp_path / "default-uv"
    monkeypatch.setattr(studio, "STUDIO_HOME", studio_home)
    monkeypatch.setattr(studio, "_uv_default_cache_dir", lambda cwd = None: default_cache)
    return studio_home / "cache" / "uv", default_cache


class _Result:
    returncode = 0


def _run_posix(monkeypatch, tmp_path: Path) -> dict:
    studio = _studio()
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")
    seen: dict = {}

    def _fake_run(
        argv,
        env = None,
        **kwargs,
    ):
        seen["argv"] = list(argv)
        seen["env"] = env
        return _Result()

    monkeypatch.setattr(studio.subprocess, "run", _fake_run)
    studio._run_setup_script(repo_root = _setup_tree(tmp_path))
    return seen


def test_the_update_uses_the_studio_cache_when_the_caller_set_none(monkeypatch, tmp_path, caches):
    studio_cache, _default = caches
    _fill(studio_cache)
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"] is not None, "env must be materialised, not left as inherit-everything"
    assert seen["env"]["UV_CACHE_DIR"] == str(studio_cache), seen["env"].get("UV_CACHE_DIR")


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_a_blank_uv_cache_dir_counts_as_unset(monkeypatch, tmp_path, caches, blank):
    """storage_roots.py:373 treats blank as unset; the update path must agree, or an
    inherited UV_CACHE_DIR= pins uv's cache to the empty string."""
    studio_cache, _default = caches
    _fill(studio_cache)
    monkeypatch.setenv("UV_CACHE_DIR", blank)
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"]["UV_CACHE_DIR"] == str(studio_cache), seen["env"].get("UV_CACHE_DIR")


def test_an_explicit_uv_cache_dir_still_wins(monkeypatch, tmp_path, caches):
    """Same precedence the installers use: a nonblank caller value is preserved
    (install.sh:626, install.ps1:1232), so CI images that pin a cache keep it."""
    studio_cache, _default = caches
    _fill(studio_cache)
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "caller cache"))
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"] is None or seen["env"]["UV_CACHE_DIR"] == str(
        tmp_path / "caller cache"
    ), seen["env"]


def test_verbose_keeps_its_own_flag_alongside_the_cache(monkeypatch, tmp_path, caches):
    """The verbose branch builds env first; the cache seeding must extend it, not
    replace it."""
    studio = _studio()
    studio_cache, _default = caches
    _fill(studio_cache)
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")
    seen: dict = {}

    def _fake_run(
        argv,
        env = None,
        **kwargs,
    ):
        seen["env"] = env
        return _Result()

    monkeypatch.setattr(studio.subprocess, "run", _fake_run)
    studio._run_setup_script(verbose = True, repo_root = _setup_tree(tmp_path))

    assert seen["env"]["UNSLOTH_VERBOSE"] == "1", seen["env"].get("UNSLOTH_VERBOSE")
    assert seen["env"]["UV_CACHE_DIR"] == str(studio_cache)


def test_the_windows_branch_gets_the_same_cache(monkeypatch, tmp_path, caches):
    """setup.ps1 runs the same uv pip installs, so the PowerShell spawn needs it too."""
    studio = _studio()
    studio_cache, _default = caches
    _fill(studio_cache)
    monkeypatch.setattr(studio.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        studio._studio_runtime_gate, "resolve_windows_powershell", lambda: "powershell.exe"
    )
    monkeypatch.setattr(studio, "_probe_profile_proxy_defaults", lambda hosts: None)
    monkeypatch.setattr(studio, "_wait_for_windows_setup_process", lambda process: 0)
    seen: dict = {}

    class _Process:
        pass

    def _fake_popen(
        argv,
        env = None,
        **kwargs,
    ):
        seen["env"] = env
        return _Process()

    monkeypatch.setattr(studio.subprocess, "Popen", _fake_popen)
    studio._run_setup_script(repo_root = _setup_tree(tmp_path))

    assert seen["env"]["UV_CACHE_DIR"] == str(studio_cache)


def test_the_seeding_does_not_leak_into_this_process(monkeypatch, tmp_path, caches):
    """_ensure_studio_env_exported mutates os.environ on purpose; this must not, or a
    later `unsloth studio` in the same process would look like a caller override to
    storage_roots._setup_cache_env."""
    studio_cache, _default = caches
    _fill(studio_cache)
    _run_posix(monkeypatch, tmp_path)

    assert "UV_CACHE_DIR" not in os.environ, os.environ.get("UV_CACHE_DIR")


# --- Do not move the update off a warm cache onto a cold one -------------------------


def test_a_shared_mode_install_keeps_the_cache_that_actually_has_the_wheels(
    monkeypatch, tmp_path, caches
):
    """install.sh picks uv's own cache when it is already populated, and
    _setup_cache_env mkdirs an empty Studio cache on every server start. Redirecting
    here would re-download online and, under UV_OFFLINE / `offline = true`, fail:
    uv reads only what is cached, and nothing is."""
    _studio_cache, default_cache = caches
    _fill(default_cache)
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"]["UV_CACHE_DIR"] == str(default_cache), seen["env"].get("UV_CACHE_DIR")


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_a_blank_value_is_replaced_even_when_the_default_cache_wins(
    monkeypatch, tmp_path, caches, blank
):
    """Leaving the environment alone here would hand uv the blank value it inherited.
    uv reads UV_CACHE_DIR as --cache-dir, and `UV_CACHE_DIR= uv cache dir` exits 2 with
    "a value is required for '--cache-dir'", so every uv call in setup.sh would die."""
    _studio_cache, default_cache = caches
    _fill(default_cache)
    monkeypatch.setenv("UV_CACHE_DIR", blank)
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"]["UV_CACHE_DIR"] == str(default_cache), seen["env"].get("UV_CACHE_DIR")


def test_the_chosen_cache_is_named_rather_than_left_to_the_child(monkeypatch, tmp_path, caches):
    """setup.sh changes directory before running uv, so a cache uv resolved from a
    uv.toml beside the caller is not one the child would find again. Whichever cache
    wins, it reaches the child as an explicit path."""
    studio_cache, default_cache = caches
    for warm, expected in ((default_cache, default_cache), (studio_cache, studio_cache)):
        # Each iteration states its own starting point: the other cache goes cold, and the
        # previous run's backfilled marker goes away, or either would decide this one.
        for cache in (studio_cache, default_cache):
            shutil.rmtree(cache, ignore_errors = True)
        _fill(warm)
        (tmp_path / "StudioHome" / "cache" / "uv-cache-dir").unlink(missing_ok = True)
        seen = _run_posix(monkeypatch, tmp_path)
        assert seen["env"]["UV_CACHE_DIR"] == str(expected), seen["env"].get("UV_CACHE_DIR")


def test_a_studio_mode_install_wins_over_a_cold_default(monkeypatch, tmp_path, caches):
    """The case the PR exists for: the installer filled the Studio cache and the update
    was downloading all of it again into a cache holding nothing."""
    studio_cache, _default = caches
    _fill(studio_cache)
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"]["UV_CACHE_DIR"] == str(studio_cache), seen["env"].get("UV_CACHE_DIR")


def test_two_warm_caches_and_no_marker_keep_uv_s_default(monkeypatch, tmp_path, caches):
    """An install that predates the marker cannot say which cache it used, and content
    cannot tell either: one on-demand wheel warms the Studio cache (install.sh:705). uv's
    default is what it has been updating from, and it cannot record its way out."""
    studio_cache, default_cache = caches
    _fill(studio_cache)
    _fill(default_cache)
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"]["UV_CACHE_DIR"] == str(default_cache), seen["env"].get("UV_CACHE_DIR")


def test_a_marker_still_settles_two_warm_caches(monkeypatch, tmp_path, caches):
    """And that ambiguity is exactly what the marker removes: an install that recorded the
    Studio cache keeps it, warm default or not."""
    studio_cache, default_cache = caches
    _fill(studio_cache)
    _fill(default_cache)
    _record(tmp_path / "StudioHome", studio_cache)
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"]["UV_CACHE_DIR"] == str(studio_cache), seen["env"].get("UV_CACHE_DIR")


def test_two_cold_caches_still_go_to_the_studio_one(monkeypatch, tmp_path, caches):
    """Nothing to preserve, so keep the bytes under the Studio root where uninstall can
    reclaim them (#10193)."""
    studio_cache, _default = caches
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"]["UV_CACHE_DIR"] == str(studio_cache), seen["env"].get("UV_CACHE_DIR")


@pytest.mark.parametrize("metadata", ["resolve.msgpack", "wheel.http", "pkg.lock", "x.rev"])
def test_a_metadata_only_default_cache_is_not_warm(monkeypatch, tmp_path, caches, metadata):
    """wheels-v6 holds only metadata on uv 0.10, so a single `uv pip install --dry-run`
    would otherwise pin every later update to uv's default cache (#10204)."""
    studio_cache, default_cache = caches
    _fill(default_cache, bucket = "wheels-v6", name = metadata)
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"]["UV_CACHE_DIR"] == str(studio_cache), seen["env"].get("UV_CACHE_DIR")


def test_a_default_cache_uv_cannot_name_does_not_block_the_redirect(monkeypatch, tmp_path, caches):
    """No uv on PATH, or `uv cache dir` failing, must not strand the update on a cache
    nobody can identify."""
    studio = _studio()
    studio_cache, _default = caches
    monkeypatch.setattr(studio, "_uv_default_cache_dir", lambda cwd = None: None)
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"]["UV_CACHE_DIR"] == str(studio_cache), seen["env"].get("UV_CACHE_DIR")


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_no_cache_mode_neither_seeds_nor_records(monkeypatch, tmp_path, caches, value):
    """uv --no-cache caches in a temporary directory and discards it at exit, so seeding
    a cache would be ignored and recording one would name something already gone."""
    studio = _studio()
    studio_cache, default_cache = caches
    _fill(default_cache)
    monkeypatch.setenv("UV_NO_CACHE", value)
    monkeypatch.setattr(
        studio,
        "_uv_default_cache_dir",
        lambda cwd = None: pytest.fail("the probe ran under --no-cache"),
    )
    seen = _run_posix(monkeypatch, tmp_path)

    assert "UV_CACHE_DIR" not in (seen["env"] or {}), seen["env"]
    assert not _marker(tmp_path).exists()


@pytest.mark.parametrize("value", ["0", "false", "", "maybe"])
def test_a_non_true_no_cache_value_changes_nothing(monkeypatch, tmp_path, caches, value):
    """0 and false leave caching on, and an unparseable value is one uv refuses to run
    on at all."""
    studio_cache, _default = caches
    _fill(studio_cache)
    monkeypatch.setenv("UV_NO_CACHE", value)
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"]["UV_CACHE_DIR"] == str(studio_cache), seen["env"].get("UV_CACHE_DIR")


# --- The warmth test itself ----------------------------------------------------------


def test_package_bytes_beside_metadata_count_as_warm(tmp_path):
    """A real cache has both; the metadata filter must not hide the payload."""
    studio = _studio()
    cache = tmp_path / "uv"
    _fill(cache, bucket = "wheels-v6", name = "resolve.msgpack")
    _fill(cache, bucket = "wheels-v6", name = "torch.whl")

    assert studio._uv_cache_has_packages(cache) is True


@pytest.mark.parametrize("bucket", ["archive-v0", "builds-v0", "built-wheels-v3", "sdists-v9"])
def test_every_bucket_uv_has_used_counts(tmp_path, bucket):
    """uv renames these across versions, and an install.sh that scans a bucket this does
    not would disagree with itself about which cache is warm."""
    studio = _studio()
    cache = tmp_path / bucket.split("-")[0]
    _fill(cache, bucket = bucket)

    assert studio._uv_cache_has_packages(cache) is True


def test_an_absent_cache_is_not_warm(tmp_path):
    studio = _studio()

    assert studio._uv_cache_has_packages(tmp_path / "nope") is False


# --- The cache the installer recorded --------------------------------------------------


def _record(studio_home: Path, value) -> None:
    marker = studio_home / "cache"
    marker.mkdir(parents = True, exist_ok = True)
    (marker / "uv-cache-dir").write_text(f"{value}\n", encoding = "utf-8")


def test_the_recorded_install_cache_beats_both_guesses(monkeypatch, tmp_path, caches):
    """The case content cannot decide: a shared install whose backend has since dropped
    one wheel into the Studio cache (install.sh:705), so both caches hold packages."""
    studio_cache, default_cache = caches
    _fill(studio_cache, name = "some_runtime_wheel.whl")
    _fill(default_cache)
    _record(tmp_path / "StudioHome", default_cache)
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"]["UV_CACHE_DIR"] == str(default_cache), seen["env"].get("UV_CACHE_DIR")


def test_a_recorded_studio_cache_survives_the_user_warming_their_own(monkeypatch, tmp_path, caches):
    """The mirror: a studio install, and the user has since warmed uv's own cache."""
    studio_cache, default_cache = caches
    _fill(studio_cache)
    _fill(default_cache)
    _record(tmp_path / "StudioHome", studio_cache)
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"]["UV_CACHE_DIR"] == str(studio_cache), seen["env"].get("UV_CACHE_DIR")


def test_an_emptied_recorded_cache_does_not_outrank_a_warm_one(monkeypatch, tmp_path, caches):
    """A marker pointing at an emptied cache is stale, not authoritative."""
    studio_cache, default_cache = caches
    _fill(default_cache)
    _record(tmp_path / "StudioHome", studio_cache)
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"]["UV_CACHE_DIR"] == str(default_cache), seen["env"].get("UV_CACHE_DIR")


def test_installs_older_than_the_marker_still_work(monkeypatch, tmp_path, caches):
    """The normal state for everyone installed before this change."""
    studio_cache, _default = caches
    _fill(studio_cache)
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"]["UV_CACHE_DIR"] == str(studio_cache), seen["env"].get("UV_CACHE_DIR")


def test_a_reinstall_into_a_custom_cache_is_not_shadowed_by_the_old_marker(
    monkeypatch, tmp_path, caches
):
    """A reinstall with a nonblank UV_CACHE_DIR fills that cache, so it is recorded too:
    a stale marker would aim later updates at a cache this install never filled, and both
    hold packages, so nothing downstream could notice."""
    studio_cache, _default = caches
    custom = tmp_path / "caller cache"
    _fill(studio_cache)
    _fill(custom)
    _record(tmp_path / "StudioHome", custom)
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"]["UV_CACHE_DIR"] == str(custom), seen["env"].get("UV_CACHE_DIR")


@pytest.mark.parametrize("spelling", ["trailing ", " leading", "  both  "])
def test_a_recorded_path_keeps_its_whitespace(monkeypatch, tmp_path, caches, spelling):
    """A recorded name may start or end with a space, and stripping it would probe a
    different path and read a warm cache as cold."""
    _studio_cache, _default = caches
    odd = tmp_path / spelling
    _fill(odd)
    _record(tmp_path / "StudioHome", odd)
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"]["UV_CACHE_DIR"] == str(odd), seen["env"].get("UV_CACHE_DIR")


def test_a_marker_written_by_windows_powershell_is_read_back(monkeypatch, tmp_path, caches):
    """PowerShell 5.1 writes `-Encoding utf8` with a BOM, which utf-8 would decode into
    the first character of the path."""
    studio_cache, default_cache = caches
    _fill(studio_cache)
    _fill(default_cache)
    marker = tmp_path / "StudioHome" / "cache"
    marker.mkdir(parents = True, exist_ok = True)
    (marker / "uv-cache-dir").write_bytes(b"\xef\xbb\xbf" + f"{studio_cache}\r\n".encode("utf-8"))
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"]["UV_CACHE_DIR"] == str(studio_cache), seen["env"].get("UV_CACHE_DIR")


def test_a_relative_marker_is_resolved_before_it_is_handed_over(monkeypatch, tmp_path, caches):
    """setup.sh changes directory, so a relative path would name somewhere else there."""
    studio_cache, _default = caches
    _fill(studio_cache)
    monkeypatch.chdir(tmp_path)
    _record(tmp_path / "StudioHome", "StudioHome/cache/uv")
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"]["UV_CACHE_DIR"] == str(studio_cache), seen["env"].get("UV_CACHE_DIR")


def test_a_recorded_path_containing_a_newline_survives_the_round_trip(
    monkeypatch, tmp_path, caches
):
    """A newline is legal in a POSIX path. Read line by line the record looked like
    several, only the last fragment survived, and a warm cache read as absent."""
    studio = _studio()
    studio_cache, _default = caches
    _fill(studio_cache)
    weird = tmp_path / "two\nline cache"
    _fill(weird)
    _record(tmp_path / "StudioHome", weird)

    env = studio._with_studio_uv_cache({})

    assert env["UV_CACHE_DIR"] == str(weird)


# --- Backfilling the marker for installs that predate it --------------------------------


def _marker(tmp_path: Path) -> Path:
    return tmp_path / "StudioHome" / "cache" / "uv-cache-dir"


def test_a_legacy_install_records_what_the_update_worked_out(monkeypatch, tmp_path, caches):
    """Otherwise the fallback is re-derived every time, and goes stale the moment the
    backend drops one wheel into the Studio cache."""
    _studio_cache, default_cache = caches
    _fill(default_cache)
    _run_posix(monkeypatch, tmp_path)

    assert _marker(tmp_path).read_text(encoding = "utf-8").strip() == str(default_cache)


def test_a_failed_update_records_nothing(monkeypatch, tmp_path, caches):
    """A cache that did not get through setup is not one to aim later updates at."""
    studio = _studio()
    _studio_cache, default_cache = caches
    _fill(default_cache)
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")

    class _Failed:
        returncode = 1

    monkeypatch.setattr(studio.subprocess, "run", lambda argv, **kw: _Failed())
    with pytest.raises(studio.typer.Exit):
        studio._run_setup_script(repo_root = _setup_tree(tmp_path))

    assert not _marker(tmp_path).exists(), _marker(tmp_path).read_text(encoding = "utf-8")


def test_a_live_marker_is_not_overwritten_by_the_update(monkeypatch, tmp_path, caches):
    """The installer's statement outranks the update's inference while it still holds."""
    studio_cache, _default = caches
    _fill(studio_cache)
    _record(tmp_path / "StudioHome", studio_cache)
    _run_posix(monkeypatch, tmp_path)

    assert _marker(tmp_path).read_text(encoding = "utf-8").strip() == str(studio_cache)


def test_a_stale_marker_is_replaced_once_the_fallback_works(monkeypatch, tmp_path, caches):
    """A marker whose cache was emptied is already ignored; stop re-deriving that."""
    studio_cache, default_cache = caches
    _fill(default_cache)
    _record(tmp_path / "StudioHome", studio_cache)
    _run_posix(monkeypatch, tmp_path)

    assert _marker(tmp_path).read_text(encoding = "utf-8").strip() == str(default_cache)


def test_a_caller_supplied_cache_is_never_promoted_to_the_marker(monkeypatch, tmp_path, caches):
    """One run's environment variable must not become every later update's default."""
    studio_cache, _default = caches
    _fill(studio_cache)
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "caller cache"))
    _run_posix(monkeypatch, tmp_path)

    assert not _marker(tmp_path).exists()


def test_a_staged_update_parks_its_choice_in_the_stage(monkeypatch, tmp_path, caches):
    """STUDIO_HOME names the LIVE install even in a staged child and the stage can still
    be rejected, so the choice waits there for stage() to promote it. Dropping it left
    desktop-only installs on the content fallback forever."""
    studio = _studio()
    _studio_cache, default_cache = caches
    _fill(default_cache)
    stage = tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setenv(studio._studio_stage.STAGE_ROOT_ENV, str(stage))
    _run_posix(monkeypatch, tmp_path)

    assert not _marker(tmp_path).exists()
    assert (stage / "uv-cache-dir").read_text(encoding = "utf-8").strip() == str(default_cache)


def test_a_marker_holding_a_null_byte_reads_as_a_cold_cache(monkeypatch, tmp_path, caches):
    """Path takes an embedded NUL and scandir then raises ValueError, not OSError, so a
    corrupted or hand-edited marker aborted every update before setup ran."""
    studio = _studio()
    studio_cache, _default = caches
    _fill(studio_cache)
    _record(tmp_path / "StudioHome", "/nul\x00cache")

    env = studio._with_studio_uv_cache({})

    assert env["UV_CACHE_DIR"] == str(studio_cache)


def test_the_backfill_replaces_a_symlink_rather_than_its_target(monkeypatch, tmp_path, caches):
    """write_text follows a symlink, so a marker path linked elsewhere would truncate an
    unrelated file."""
    _studio_cache, default_cache = caches
    _fill(default_cache)
    victim = tmp_path / "someone elses file"
    victim.write_text("do not clobber", encoding = "utf-8")
    marker = _marker(tmp_path)
    marker.parent.mkdir(parents = True, exist_ok = True)
    marker.symlink_to(victim)
    _run_posix(monkeypatch, tmp_path)

    assert victim.read_text(encoding = "utf-8") == "do not clobber"
    assert not marker.is_symlink()
    assert marker.read_text(encoding = "utf-8").strip() == str(default_cache)


@pytest.mark.skipif(os.name != "posix", reason = "POSIX filesystem byte semantics")
def test_a_cache_path_that_is_not_utf_8_is_recorded_and_read_back(monkeypatch, tmp_path):
    """An undecodable POSIX path arrives as surrogates, and encoding those raises
    UnicodeEncodeError, not an OSError, so it escaped the best-effort handler and failed
    an update whose setup had succeeded, with its marker already gone."""
    studio = _studio()
    weird = (tmp_path / os.fsdecode(b"caf\xe9-cache")).resolve()
    _fill(weird)
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path / "StudioHome")
    monkeypatch.setattr(studio, "_uv_default_cache_dir", lambda cwd = None: weird)
    monkeypatch.delenv("UV_CACHE_DIR", raising = False)

    studio._backfill_uv_cache_marker({"UV_CACHE_DIR": str(weird)})

    assert _marker(tmp_path).read_bytes().strip() == os.fsencode(str(weird))
    # And the reader gives back the path the filesystem uses, not one with U+FFFD in it.
    assert studio._recorded_install_uv_cache() == weird
    # The tmp_path reaper cannot always delete a name it cannot decode.
    shutil.rmtree(weird, ignore_errors = True)


# --- The uv probe ---------------------------------------------------------------------


def _probe_kwargs(
    monkeypatch,
    stdout: str = "/cache/uv\n",
    cwd = None,
) -> dict:
    studio = _studio()
    monkeypatch.setattr(studio.shutil, "which", lambda name: "/usr/bin/uv")
    seen: dict = {}

    class _Completed:
        returncode = 0

    def _fake_run(argv, **kwargs):
        seen.update(kwargs)
        seen["argv"] = list(argv)
        completed = _Completed()
        completed.stdout = stdout
        return completed

    monkeypatch.setattr(studio.subprocess, "run", _fake_run)
    seen["result"] = studio._uv_default_cache_dir(cwd)
    return seen


def test_the_probe_asks_from_the_directory_setup_will_ask_from(monkeypatch, tmp_path):
    """uv discovers uv.toml and pyproject.toml from its working directory, and both setup
    scripts change into their own before the dependency pass (studio/setup.sh:1788). Asked
    in the caller's directory instead, the probe answers for whatever project the user
    happens to be standing in, and that answer is then forced on the child."""
    seen = _probe_kwargs(monkeypatch, stdout = "relcache\n", cwd = tmp_path / "studio")

    assert seen["cwd"] == str(tmp_path / "studio")
    # And the relative answer resolves against that directory, not against the caller's.
    assert seen["result"] == tmp_path / "studio" / "relcache", seen["result"]


def test_the_windows_handoff_probes_from_the_directory_it_hands_the_child(monkeypatch, tmp_path):
    """setup.ps1 never changes directory: it hands install_python_stack.py the cwd it
    inherited (setup.ps1:5191), so probing the script's directory would answer for a
    configuration the child never sees, and that answer is forced on it."""
    studio = _studio()
    repo_root = tmp_path / "repo"
    (repo_root / "studio").mkdir(parents = True)
    (repo_root / "studio" / "setup.ps1").write_text("", encoding = "utf-8")
    monkeypatch.setattr(studio.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        studio._studio_runtime_gate, "resolve_windows_powershell", lambda: "powershell.exe"
    )
    monkeypatch.setattr(studio, "_probe_profile_proxy_defaults", lambda hosts: None)
    monkeypatch.setattr(studio, "_backfill_uv_cache_marker", lambda env: None)
    monkeypatch.delenv("UV_CACHE_DIR", raising = False)
    seen: dict = {}
    monkeypatch.setattr(
        studio, "_with_studio_uv_cache", lambda env, cwd = None: seen.update(cwd = cwd) or env
    )

    class _Process:
        def wait(self):
            return 0

    monkeypatch.setattr(studio.subprocess, "Popen", lambda argv, **kw: _Process())

    studio._run_setup_script(repo_root = repo_root)

    assert seen["cwd"] is None, seen


def test_the_posix_handoff_probes_from_the_setup_script_directory(monkeypatch, tmp_path):
    """setup.sh does change into its own directory before the dependency pass
    (studio/setup.sh:1788), so there is where the probe has to ask."""
    studio = _studio()
    repo_root = tmp_path / "repo"
    (repo_root / "studio").mkdir(parents = True)
    (repo_root / "studio" / "setup.sh").write_text("", encoding = "utf-8")
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")
    monkeypatch.setattr(studio, "_backfill_uv_cache_marker", lambda env: None)
    monkeypatch.delenv("UV_CACHE_DIR", raising = False)
    seen: dict = {}
    monkeypatch.setattr(
        studio, "_with_studio_uv_cache", lambda env, cwd = None: seen.update(cwd = cwd) or env
    )
    monkeypatch.setattr(
        studio.subprocess, "Popen", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("stop"))
    )

    with pytest.raises(RuntimeError, match = "stop"):
        studio._run_setup_script(repo_root = repo_root)

    assert seen["cwd"] == repo_root / "studio", seen


def test_the_probe_keeps_whitespace_uv_reported(monkeypatch):
    """A directory name may begin or end with a space and uv reports it verbatim, so
    stripping the line probes a path that does not exist and reads a warm cache as cold."""
    seen = _probe_kwargs(monkeypatch, stdout = "/spaced cache/uv  \n")

    assert seen["result"] == Path("/spaced cache/uv  "), seen["result"]


def test_a_relative_uv_working_dir_anchors_to_the_probe_directory(monkeypatch, tmp_path):
    """uv resolves --directory after starting where it was started, so a relative
    UV_WORKING_DIR belongs to the probe's cwd, not to wherever the update was launched."""
    monkeypatch.setenv("UV_WORKING_DIR", "uvdir")
    seen = _probe_kwargs(monkeypatch, stdout = "relcache\n", cwd = tmp_path / "studio")

    assert seen["result"] == tmp_path / "studio" / "uvdir" / "relcache", seen["result"]


def test_the_probe_decodes_utf8_whatever_the_console_codec_is(monkeypatch):
    """text=True alone decodes with the locale codec and strict errors, so a non-ASCII
    cache path raises UnicodeDecodeError. That is a ValueError, so the OSError /
    SubprocessError handler does not catch it and the update dies. Same reason the
    profile probe above already pins the codec."""
    seen = _probe_kwargs(monkeypatch)

    assert seen["encoding"] == "utf-8", seen.get("encoding")
    assert seen["errors"] == "replace", seen.get("errors")


def test_the_probe_is_hidden_like_every_other_spawn(monkeypatch):
    """A desktop update runs the CLI with CREATE_NO_WINDOW, and creation flags are not
    inherited, so each process it starts has to ask for them itself."""
    studio = _studio()
    monkeypatch.setattr(studio, "_should_hide_windows_subprocesses", lambda: True)
    monkeypatch.setattr(studio.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising = False)
    seen = _probe_kwargs(monkeypatch)

    expected = studio._windows_hidden_subprocess_kwargs()
    for key, value in expected.items():
        assert key in seen, f"{key} missing from the probe call"
    assert seen.get("creationflags") == expected.get("creationflags")


def test_the_probe_absolutises_a_relative_answer(monkeypatch, tmp_path):
    """uv answers `cache-dir = "relcache"` with "relcache" verbatim, and setup.sh runs uv
    from a different directory, so a relative answer names a different, cold cache there."""
    monkeypatch.delenv("UV_WORKING_DIR", raising = False)
    monkeypatch.chdir(tmp_path)
    seen = _probe_kwargs(monkeypatch, stdout = "relcache\n")

    assert seen["result"] == tmp_path / "relcache", seen["result"]


def test_the_probe_resolves_against_uvs_working_directory(monkeypatch, tmp_path):
    """--directory, whose env alias is UV_WORKING_DIR, moves uv out from under us before
    it resolves a relative cache-dir. Measured on uv 0.10.7: with UV_WORKING_DIR set, the
    cache lands under that directory and not under ours."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv("UV_WORKING_DIR", str(elsewhere))
    monkeypatch.chdir(tmp_path)
    seen = _probe_kwargs(monkeypatch, stdout = "relcache\n")

    assert seen["result"] == elsewhere / "relcache", seen["result"]


def test_the_probe_leaves_uvs_tilde_alone(monkeypatch, tmp_path):
    """uv prints `cache-dir = "~/.myuv"` verbatim and treats the tilde as an ordinary
    relative segment: measured on uv 0.10.7 it creates a literal "~" directory in its
    working directory. Expanding it here would probe a path uv never writes to."""
    monkeypatch.delenv("UV_WORKING_DIR", raising = False)
    monkeypatch.chdir(tmp_path)
    seen = _probe_kwargs(monkeypatch, stdout = "~/.myuv\n")

    assert seen["result"] == tmp_path / "~" / ".myuv", seen["result"]


def test_a_probe_that_blows_up_costs_a_preference_not_the_update(monkeypatch, tmp_path):
    """subprocess.run is implemented with Popen, so any caller that fakes Popen reaches
    this probe too. Raising through an update is not acceptable, and neither is reporting
    no cache at all: it falls back to the platform default install.sh:646 uses."""
    studio = _studio()
    monkeypatch.setattr(studio.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    def _boom(argv, **kwargs):
        raise TypeError("object does not support the context manager protocol")

    monkeypatch.setattr(studio.subprocess, "run", _boom)

    assert studio._uv_default_cache_dir() == tmp_path / "uv"


def test_a_malformed_uv_toml_beside_the_caller_does_not_hide_the_cache(monkeypatch, tmp_path):
    """uv discovers config from the CURRENT directory, so a broken uv.toml where the user
    happens to be makes `uv cache dir` exit nonzero even though setup.sh changes directory
    before it runs uv. install.sh falls back to the platform default rather than calling
    the cache cold, and so must this."""
    studio = _studio()
    monkeypatch.setattr(studio.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    class _Failed:
        returncode = 2
        stdout = ""

    monkeypatch.setattr(studio.subprocess, "run", lambda argv, **kw: _Failed())

    assert studio._uv_default_cache_dir() == tmp_path / "uv"


@pytest.mark.parametrize(
    ("system", "env", "expected"),
    [
        ("Linux", {"XDG_CACHE_HOME": "/xdg"}, Path("/xdg/uv")),
        ("Linux", {"HOME": "/home/someone"}, Path("/home/someone/.cache/uv")),
        ("Darwin", {"HOME": "/Users/someone"}, Path("/Users/someone/.cache/uv")),
        ("Windows", {"LOCALAPPDATA": "/local"}, Path("/local/uv/cache")),
    ],
)
def test_the_platform_default_matches_the_installers(monkeypatch, system, env, expected):
    """Same locations install.sh:646 and install.ps1 use. macOS is XDG/HOME, not
    ~/Library/Caches, which is what uv documents for Unix generally."""
    studio = _studio()
    monkeypatch.setattr(studio.platform, "system", lambda: system)
    for key in ("XDG_CACHE_HOME", "HOME", "LOCALAPPDATA"):
        monkeypatch.delenv(key, raising = False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    assert studio._uv_platform_cache_dir() == expected


def test_no_platform_default_is_better_than_a_guess(monkeypatch):
    """Nothing to anchor to means nothing to report; the caller then keeps the Studio
    cache rather than probing a path assembled out of nothing."""
    studio = _studio()
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")
    for key in ("XDG_CACHE_HOME", "HOME"):
        monkeypatch.delenv(key, raising = False)

    assert studio._uv_platform_cache_dir() is None


def test_the_probe_reads_the_last_nonblank_line(monkeypatch):
    """A wrapper or a future uv may print a notice ahead of the path, and a two-line
    value is not a directory."""
    seen = _probe_kwargs(monkeypatch, stdout = "warning: something\n/real/cache/uv\n\n")

    assert seen["result"] == Path("/real/cache/uv"), seen["result"]


def test_the_probe_hides_uv_cache_dir_from_uv(monkeypatch):
    """Asking uv for its default while UV_CACHE_DIR is set would just echo that value
    back, and a blank one makes uv exit 2 instead of answering."""
    monkeypatch.setenv("UV_CACHE_DIR", "")
    seen = _probe_kwargs(monkeypatch)

    assert "UV_CACHE_DIR" not in seen["env"], seen["env"].get("UV_CACHE_DIR")


def test_files_outside_a_bucket_are_not_warm(tmp_path):
    """uv writes CACHEDIR.TAG and .gitignore at the cache root; a cache holding only
    those has never fetched anything."""
    studio = _studio()
    cache = tmp_path / "uv"
    cache.mkdir()
    (cache / "CACHEDIR.TAG").write_text("Signature: 8a477f597d28d172789f06886806bc55")
    (cache / ".gitignore").write_text("*")
    (cache / "simple-v20").mkdir()
    (cache / "simple-v20" / "index.msgpack").write_bytes(b"\0")

    assert studio._uv_cache_has_packages(cache) is False

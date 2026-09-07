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
    monkeypatch.setattr(studio, "_uv_default_cache_dir", lambda: default_cache)
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
        _fill(warm)
        # The previous iteration's run backfills a marker, which would then decide this
        # one. Each iteration states its own starting point.
        (tmp_path / "StudioHome" / "cache" / "uv-cache-dir").unlink(missing_ok = True)
        seen = _run_posix(monkeypatch, tmp_path)
        assert seen["env"]["UV_CACHE_DIR"] == str(expected), seen["env"].get("UV_CACHE_DIR")


def test_a_studio_mode_install_wins_over_a_populated_default(monkeypatch, tmp_path, caches):
    """The case the PR exists for: the installer filled the Studio cache, so the update
    must read it rather than uv's default, whatever else is lying around."""
    studio_cache, default_cache = caches
    _fill(studio_cache)
    _fill(default_cache)
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
    monkeypatch.setattr(studio, "_uv_default_cache_dir", lambda: None)
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
    """The case content cannot decide: a shared-mode install, and the running backend has
    since dropped one wheel into the Studio cache (install.sh:705 points it there even in
    shared mode), so both caches hold package bytes."""
    studio_cache, default_cache = caches
    _fill(studio_cache, name = "some_runtime_wheel.whl")
    _fill(default_cache)
    _record(tmp_path / "StudioHome", default_cache)
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"]["UV_CACHE_DIR"] == str(default_cache), seen["env"].get("UV_CACHE_DIR")


def test_a_recorded_studio_cache_survives_the_user_warming_their_own(monkeypatch, tmp_path, caches):
    """The mirror image: a studio-mode install, and the user has since used uv for
    something else. Content alone would hand the update a cache the install never used."""
    studio_cache, default_cache = caches
    _fill(studio_cache)
    _fill(default_cache)
    _record(tmp_path / "StudioHome", studio_cache)
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"]["UV_CACHE_DIR"] == str(studio_cache), seen["env"].get("UV_CACHE_DIR")


def test_an_emptied_recorded_cache_does_not_outrank_a_warm_one(monkeypatch, tmp_path, caches):
    """`uv cache clean` is the user's to run. A marker pointing at nothing is stale, not
    authoritative."""
    studio_cache, default_cache = caches
    _fill(default_cache)
    _record(tmp_path / "StudioHome", studio_cache)
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"]["UV_CACHE_DIR"] == str(default_cache), seen["env"].get("UV_CACHE_DIR")


def test_installs_older_than_the_marker_still_work(monkeypatch, tmp_path, caches):
    """No marker is the normal state for everyone installed before this change, so the
    content fallback has to stay."""
    studio_cache, _default = caches
    _fill(studio_cache)
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"]["UV_CACHE_DIR"] == str(studio_cache), seen["env"].get("UV_CACHE_DIR")


def test_a_reinstall_into_a_custom_cache_is_not_shadowed_by_the_old_marker(
    monkeypatch, tmp_path, caches
):
    """A reinstall with a nonblank UV_CACHE_DIR fills that cache, so the installers record
    it too. Were the previous install's marker left behind, a later update without the
    variable would read a cache this install never filled, and both still hold packages,
    so nothing downstream could notice."""
    studio_cache, _default = caches
    custom = tmp_path / "caller cache"
    _fill(studio_cache)
    _fill(custom)
    _record(tmp_path / "StudioHome", custom)
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"]["UV_CACHE_DIR"] == str(custom), seen["env"].get("UV_CACHE_DIR")


@pytest.mark.parametrize("spelling", ["trailing ", " leading", "  both  "])
def test_a_recorded_path_keeps_its_whitespace(monkeypatch, tmp_path, caches, spelling):
    """The installers write UV_CACHE_DIR through verbatim and a directory name may
    legitimately start or end with a space, so stripping the line would probe a different
    path and read a warm cache as cold."""
    _studio_cache, _default = caches
    odd = tmp_path / spelling
    _fill(odd)
    _record(tmp_path / "StudioHome", odd)
    seen = _run_posix(monkeypatch, tmp_path)

    assert seen["env"]["UV_CACHE_DIR"] == str(odd), seen["env"].get("UV_CACHE_DIR")


def test_a_marker_written_by_windows_powershell_is_read_back(monkeypatch, tmp_path, caches):
    """Windows PowerShell 5.1 writes `-Encoding utf8` with a BOM, and utf-8 would decode
    it into the first character of the path."""
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


# --- Backfilling the marker for installs that predate it --------------------------------


def _marker(tmp_path: Path) -> Path:
    return tmp_path / "StudioHome" / "cache" / "uv-cache-dir"


def test_a_legacy_install_records_what_the_update_worked_out(monkeypatch, tmp_path, caches):
    """Otherwise the fallback has to be re-derived every time, and it goes stale the
    moment the backend drops one wheel into the Studio cache."""
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
    """`uv cache clean` on the recorded cache leaves a pointer to nothing; the update
    already ignores it, and should stop re-deriving that every time."""
    studio_cache, default_cache = caches
    _fill(default_cache)
    _record(tmp_path / "StudioHome", studio_cache)
    _run_posix(monkeypatch, tmp_path)

    assert _marker(tmp_path).read_text(encoding = "utf-8").strip() == str(default_cache)


def test_a_caller_supplied_cache_is_never_promoted_to_the_marker(monkeypatch, tmp_path, caches):
    """The installers record their own choice. An update must not turn one run's
    environment variable into every later update's default."""
    studio_cache, _default = caches
    _fill(studio_cache)
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "caller cache"))
    _run_posix(monkeypatch, tmp_path)

    assert not _marker(tmp_path).exists()


# --- The uv probe ---------------------------------------------------------------------


def _probe_kwargs(monkeypatch, stdout: str = "/cache/uv\n") -> dict:
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
    seen["result"] = studio._uv_default_cache_dir()
    return seen


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
    monkeypatch.chdir(tmp_path)
    seen = _probe_kwargs(monkeypatch, stdout = "relcache\n")

    assert seen["result"] == tmp_path / "relcache", seen["result"]


def test_a_probe_that_blows_up_costs_a_preference_not_the_update(monkeypatch):
    """subprocess.run is implemented with Popen, so any caller that fakes Popen reaches
    this probe too. Losing the answer is fine; raising through an update is not."""
    studio = _studio()
    monkeypatch.setattr(studio.shutil, "which", lambda name: "/usr/bin/uv")

    def _boom(argv, **kwargs):
        raise TypeError("object does not support the context manager protocol")

    monkeypatch.setattr(studio.subprocess, "run", _boom)

    assert studio._uv_default_cache_dir() is None


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

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

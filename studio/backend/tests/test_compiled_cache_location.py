# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The compiled-cache clear acts on the directory unsloth_zoo.compiler writes to and nothing else:
UNSLOTH_COMPILE_LOCATION when set, else unsloth_compiled_cache in the CWD. Never the shared temp dir the compiler
falls back to past an unusable location: anyone on the machine can create or link that name."""

import os
import tempfile
from pathlib import Path

import pytest

from utils import cache_cleanup

MODULE = "unsloth_compiled_module_llama.py"
TRAINER = "UnslothSFTTrainer.py"


def _write(path: Path, text: str = "x = 1\n") -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(text, encoding = "utf-8")
    return path


@pytest.fixture
def launch(tmp_path, monkeypatch):
    """A launch folder holding a same-named directory, and no source-tree candidates in play."""
    folder = tmp_path / "launch"
    folder.mkdir()
    monkeypatch.chdir(folder)
    monkeypatch.setattr(cache_cleanup, "_CACHE_DIRS", [])
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    monkeypatch.delenv("UNSLOTH_COMPILE_LOCATION", raising = False)
    return folder


def _pinned(tmp_path, monkeypatch) -> Path:
    pinned = tmp_path / "home" / "compiled_cache"
    _write(pinned / cache_cleanup.CACHE_MARKER, "")
    _write(pinned / MODULE)
    _write(pinned / TRAINER)
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(pinned))
    return pinned


@pytest.mark.parametrize("preserve", [["Unsloth*Trainer.py"], None])
def test_a_configured_location_leaves_the_launch_folder_alone(
    tmp_path, monkeypatch, launch, preserve
):
    stray = launch / "unsloth_compiled_cache"
    _write(stray / MODULE)
    _write(stray / "notes.txt")
    pinned = _pinned(tmp_path, monkeypatch)

    cache_cleanup.clear_unsloth_compiled_cache(preserve_patterns = preserve)

    assert (stray / MODULE).is_file(), "deleted from a folder the compiler never wrote to"
    assert (stray / "notes.txt").is_file()
    assert not (pinned / MODULE).exists()
    assert (pinned / TRAINER).is_file() is (preserve is not None)
    assert (pinned / cache_cleanup.CACHE_MARKER).is_file()


def test_a_configured_location_does_not_warn_about_the_launch_folder(tmp_path, monkeypatch, launch):
    (launch / "unsloth_compiled_cache").mkdir()
    _pinned(tmp_path, monkeypatch)
    warnings: list = []
    monkeypatch.setattr(
        cache_cleanup.logger,
        "warning",
        lambda message, *args, **_k: warnings.append(message % args),
    )

    cache_cleanup.clear_unsloth_compiled_cache(preserve_patterns = ["Unsloth*Trainer.py"])

    assert warnings == []


def test_without_the_variable_the_launch_folder_is_the_cache(launch):
    cwd_cache = launch / "unsloth_compiled_cache"
    _write(cwd_cache / MODULE)
    _write(cwd_cache / "notes.txt")

    cache_cleanup.clear_unsloth_compiled_cache(preserve_patterns = ["Unsloth*Trainer.py"])

    # Not ours by construction, so only what the compiler generated goes.
    assert not (cwd_cache / MODULE).exists()
    assert (cwd_cache / "notes.txt").is_file()


def test_a_relative_location_is_taken_against_the_cwd(tmp_path, monkeypatch, launch):
    relative = launch / "rel" / "ucc"
    _write(relative / MODULE)
    other = tmp_path / "rel" / "ucc"
    _write(other / MODULE)
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", "rel/ucc")

    cache_cleanup.clear_unsloth_compiled_cache()

    assert not (relative / MODULE).exists()
    assert (other / MODULE).is_file()


def _blocked(tmp_path, monkeypatch) -> Path:
    """A location os.makedirs cannot create, so the compiler would write to <tempdir>/ucc instead."""
    blocker = _write(tmp_path / "not_a_dir")
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(blocker / "ucc"))
    return tmp_path / "tmp" / "ucc"


@pytest.mark.parametrize("preserve", [["Unsloth*Trainer.py"], None])
def test_an_uncreatable_location_never_adopts_the_shared_temp_dir(
    tmp_path, monkeypatch, launch, preserve
):
    """<tempdir>/<leaf> is a name every user and install on the machine shares, so nothing there is provably ours."""
    fallback = _blocked(tmp_path, monkeypatch)
    _write(fallback / cache_cleanup.CACHE_MARKER, "")
    _write(fallback / MODULE)
    _write(fallback / "notes.txt")
    _write(launch / "unsloth_compiled_cache" / MODULE)

    assert cache_cleanup.get_existing_cache_dirs() == []
    cache_cleanup.clear_unsloth_compiled_cache(preserve_patterns = preserve)

    assert (fallback / MODULE).is_file()
    assert (fallback / "notes.txt").is_file()
    assert (launch / "unsloth_compiled_cache" / MODULE).is_file()


@pytest.mark.skipif(not hasattr(os, "symlink") or os.name == "nt", reason = "needs POSIX symlinks")
def test_a_temp_dir_symlink_cannot_point_the_clear_at_another_cache(tmp_path, monkeypatch, launch):
    fallback = _blocked(tmp_path, monkeypatch)
    victim = tmp_path / "victim"
    _write(victim / cache_cleanup.CACHE_MARKER, "")
    witness = _write(victim / "keep.txt")
    fallback.symlink_to(victim, target_is_directory = True)

    cache_cleanup.clear_unsloth_compiled_cache()

    assert witness.is_file()


def test_a_temp_dir_is_never_put_on_the_import_path(tmp_path, monkeypatch, launch):
    import sys

    fallback = _blocked(tmp_path, monkeypatch)
    _write(fallback / cache_cleanup.CACHE_MARKER, "")
    _write(fallback / "sitecustomize.py")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("PYTHONPATH", "")

    cache_cleanup.register_compiled_cache_on_path()

    assert str(fallback.resolve()) not in sys.path
    assert str(fallback.resolve()) not in os.environ["PYTHONPATH"]


@pytest.mark.parametrize("leaf", [".", ".."])
def test_a_dot_leaf_does_not_reach_the_temp_root(tmp_path, monkeypatch, launch, leaf):
    blocker = _write(tmp_path / "not_a_dir")
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(blocker) + "/" + leaf)
    _write(tmp_path / "tmp" / cache_cleanup.CACHE_MARKER, "")
    witness = _write(tmp_path / "tmp" / "keep.txt")

    cache_cleanup.clear_unsloth_compiled_cache()

    assert witness.is_file()


def test_a_creatable_location_never_reaches_the_temp_dir(tmp_path, monkeypatch, launch):
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(tmp_path / "not_yet" / "ucc"))
    temp_copy = _write(tmp_path / "tmp" / "ucc" / MODULE)

    assert cache_cleanup.get_existing_cache_dirs() == []
    cache_cleanup.clear_unsloth_compiled_cache()

    assert temp_copy.is_file()


@pytest.mark.parametrize("value", ["~/ucc", " ucc_padded "])
def test_the_location_is_taken_verbatim_like_the_compiler(tmp_path, monkeypatch, launch, value):
    """os.makedirs neither expands ``~`` nor strips spaces, so the compiler writes to <cwd>/~/ucc."""
    home = tmp_path / "home_dir"
    _write(home / "ucc" / MODULE)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", value)

    dirs = cache_cleanup._configured_cache_dirs()
    assert dirs == [Path.cwd() / value]
    # Absolute, so the Settings purge gate and its ``expanduser`` see the folder the compiler writes to.
    assert all(d.is_absolute() for d in dirs)
    assert str(dirs[0]) in cache_cleanup._trusted_cache_paths()

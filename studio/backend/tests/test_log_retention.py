# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep-newest-N retention for the per-session log directories.

Three things decide whether this helper is safe to point at a directory the app is
actively writing into: the cap has to mean what it says, the file the caller just opened
has to survive, and one odd directory entry must not disable retention for everything
else.
"""

import os

import pytest

from utils.log_retention import DEFAULT_KEEP, prune_log_dir

PATTERN = "llama-*.log"


def _seed(root, count):
    for i in range(count):
        path = root / f"llama-{i:04d}.log"
        path.write_text(f"log {i}\n", encoding = "utf-8")
        os.utime(path, (1_000_000 + i, 1_000_000 + i))


def _logs(root):
    return sorted(p.name for p in root.glob(PATTERN) if p.is_file())


@pytest.mark.parametrize("keep", [0, 1, 5, DEFAULT_KEEP])
def test_keep_leaves_exactly_that_many(tmp_path, keep):
    _seed(tmp_path, 30)
    prune_log_dir(tmp_path, PATTERN, keep = keep)
    assert len(_logs(tmp_path)) == keep


def test_the_newest_are_the_ones_kept(tmp_path):
    _seed(tmp_path, 30)
    prune_log_dir(tmp_path, PATTERN, keep = 3)
    assert _logs(tmp_path) == ["llama-0027.log", "llama-0028.log", "llama-0029.log"]


def test_a_negative_keep_is_a_no_op(tmp_path):
    _seed(tmp_path, 5)
    prune_log_dir(tmp_path, PATTERN, keep = -1)
    assert len(_logs(tmp_path)) == 5


def test_a_missing_directory_is_survivable(tmp_path):
    prune_log_dir(tmp_path / "not-there", PATTERN)


@pytest.mark.parametrize("population", [0, 19, 20, 21, 319])
def test_the_protected_file_counts_toward_the_cap(tmp_path, population):
    """Called after the open, the directory settles at `keep`, not `keep + 1`.

    Pruning first leaves one extra behind on every load, so the cap is never reached.
    """
    _seed(tmp_path, population)
    for load in range(5):
        active = tmp_path / f"llama-active{load}.log"
        active.write_text("live", encoding = "utf-8")
        os.utime(active, (2_000_000 + load, 2_000_000 + load))
        prune_log_dir(tmp_path, PATTERN, keep = DEFAULT_KEEP, protect = active)
        assert active.exists(), "the log this load just opened was pruned"
    assert len(_logs(tmp_path)) == min(population + 5, DEFAULT_KEEP)


def test_the_protected_file_survives_even_when_it_sorts_oldest(tmp_path):
    """Two loads in the same second, or a clock that stepped back, and the file the caller
    is writing to is no longer the newest. It still may not be deleted."""
    _seed(tmp_path, 30)
    active = tmp_path / "llama-active.log"
    active.write_text("live", encoding = "utf-8")
    os.utime(active, (1, 1))
    prune_log_dir(tmp_path, PATTERN, keep = 3, protect = active)
    assert active.exists()
    assert len(_logs(tmp_path)) == 3


def test_a_dangling_symlink_does_not_disable_retention(tmp_path):
    """stat() on a broken link raises. One try/except around the whole scan turned that
    into "never prune this directory again", the growth this helper exists to stop."""
    _seed(tmp_path, 30)
    (tmp_path / "llama-dangling.log").symlink_to(tmp_path / "gone.log")
    prune_log_dir(tmp_path, PATTERN, keep = 5)
    assert len(_logs(tmp_path)) == 5


def test_a_symlink_never_destroys_a_target_outside_the_log_directory(tmp_path):
    outside = tmp_path / "keep-me.txt"
    outside.write_text("not a log", encoding = "utf-8")
    logs = tmp_path / "logs"
    logs.mkdir()
    _seed(logs, 30)
    (logs / "llama-link.log").symlink_to(outside)
    prune_log_dir(logs, PATTERN, keep = 1)
    assert outside.read_text(encoding = "utf-8") == "not a log"


def test_a_directory_matching_the_glob_is_neither_counted_nor_removed(tmp_path):
    _seed(tmp_path, 25)
    (tmp_path / "llama-not-a-log.log").mkdir()
    prune_log_dir(tmp_path, PATTERN, keep = 5)
    assert (tmp_path / "llama-not-a-log.log").is_dir()
    assert len(_logs(tmp_path)) == 5, "a directory took one of the kept slots"


def test_identical_mtimes_still_leave_exactly_keep(tmp_path):
    for i in range(30):
        path = tmp_path / f"llama-{i:04d}.log"
        path.write_text("x", encoding = "utf-8")
        os.utime(path, (1_500_000, 1_500_000))
    prune_log_dir(tmp_path, PATTERN, keep = 7)
    assert len(_logs(tmp_path)) == 7


def test_families_do_not_prune_each_other(tmp_path):
    for i in range(30):
        (tmp_path / f"llama-{i:04d}.log").write_text("l", encoding = "utf-8")
        (tmp_path / f"diffusion-{i:04d}.log").write_text("d", encoding = "utf-8")
    prune_log_dir(tmp_path, PATTERN, keep = 5)
    assert len(list(tmp_path.glob("llama-*.log"))) == 5
    assert len(list(tmp_path.glob("diffusion-*.log"))) == 30


def test_a_writer_can_keep_appending_across_a_prune(tmp_path):
    _seed(tmp_path, 30)
    active = tmp_path / "llama-writing.log"
    with open(active, "w", encoding = "utf-8", buffering = 1) as handle:
        handle.write("first\n")
        prune_log_dir(tmp_path, PATTERN, keep = 3, protect = active)
        handle.write("second\n")
    assert active.read_text(encoding = "utf-8") == "first\nsecond\n"

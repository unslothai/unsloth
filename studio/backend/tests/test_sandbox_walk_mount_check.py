# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The host-channel walk asks ismount only where a directory's own stat can show a mount, and answers as before."""

from __future__ import annotations

import os
import random
import stat
import sys

import pytest

if os.name == "nt":
    pytest.skip("POSIX mount and permission semantics", allow_module_level = True)

from core.inference import os_sandbox  # noqa: E402


def _reference_walk(root, max_entries, witness):
    """The walk as it was: ismount for every directory, a fresh stat for every witness entry."""
    entries = 0
    links = {}
    unreadable = []
    for base, dirs, names in os.walk(
        root, followlinks = False, onerror = lambda exc: unreadable.append(exc.filename or root)
    ):
        if witness is not None:
            witness.append(os_sandbox.directory_signature(base))
        if unreadable:
            return f"{unreadable[0]} cannot be fully inspected"
        for name in (*dirs, *names):
            entries += 1
            if entries > max_entries:
                return "budget"
            path = os.path.join(base, name)
            try:
                info = os.lstat(path)
            except OSError:
                return f"changed during its safety scan: {path}"
            if stat.S_ISLNK(info.st_mode):
                continue
            if stat.S_ISDIR(info.st_mode):
                if os.path.ismount(path):
                    return f"contains a nested host mount: {path}"
                continue
            if not stat.S_ISREG(info.st_mode):
                return f"contains a device or IPC node: {path}"
            if info.st_nlink > 1:
                found = links.setdefault((info.st_dev, info.st_ino), [0, info.st_nlink, path])
                found[0] += 1
    if unreadable:
        return f"{unreadable[0]} cannot be fully inspected"
    for found, total, path in links.values():
        if found < total:
            return f"{os_sandbox._HARD_LINKED}{path}"
    return None


def _walk(
    root,
    max_entries = 50_000,
    witness = None,
):
    try:
        return os_sandbox._walk_for_host_channels(root, max_entries, 60.0, witness)
    except os_sandbox._ScanBudgetExceeded:
        return "budget"


class _Mounts:
    """Fake mounts keyed by inode, as the kernel keys them: the root shows its own device, path/.. the parent's."""

    def __init__(self, monkeypatch):
        self.points: dict[tuple[int, int], tuple[int, int | None]] = {}
        self.under: dict[tuple[int, int], int] = {}
        real_lstat, real_stat = os.lstat, os.stat

        def fake(real):
            def lookup(path, *args, **kwargs):
                info = real(path, *args, **kwargs)
                key = (info.st_dev, info.st_ino)
                if key in self.points:
                    device, inode = self.points[key]
                elif key in self.under:
                    device, inode = self.under[key], None
                else:
                    return info
                fields = list(info)
                fields[2] = device
                if inode is not None:
                    fields[1] = inode
                return os.stat_result(fields, {"st_mtime_ns": info.st_mtime_ns})

            return lookup

        monkeypatch.setattr(os, "lstat", fake(real_lstat))
        monkeypatch.setattr(os, "stat", fake(real_stat))
        self._real_stat = real_stat

    def _key(self, path):
        info = self._real_stat(path)
        return (info.st_dev, info.st_ino)

    def mount(
        self,
        path,
        device,
        inode = None,
    ):
        self.points[self._key(path)] = (device, inode)
        for base, dirs, names in os.walk(path):
            for name in (*dirs, *names):
                self.under[self._key(os.path.join(base, name))] = device


def _tree(root):
    (root / "a" / "b").mkdir(parents = True)
    (root / "a" / "b" / "f.txt").write_text("x")
    (root / "c").mkdir()
    (root / "c" / "g.txt").write_text("y")
    return root


def test_a_plain_tree_never_asks_ismount_and_witnesses_what_a_fresh_stat_reads(
    tmp_path, monkeypatch
):
    root = str(_tree(tmp_path / "root"))
    calls = []
    real = os.path.ismount
    monkeypatch.setattr(os.path, "ismount", lambda path: calls.append(path) or real(path))
    witness: list = []
    assert _walk(root, witness = witness) is None
    assert calls == []
    assert witness == [os_sandbox.directory_signature(entry[0]) for entry in witness]
    assert [entry[0] for entry in witness] == [
        base for base, _, _ in os.walk(root, followlinks = False)
    ]
    assert os_sandbox.directory_witness_matches(witness)


def test_a_child_on_another_device_is_a_nested_mount(tmp_path, monkeypatch):
    root = _tree(tmp_path / "root")
    _Mounts(monkeypatch).mount(str(root / "a" / "b"), device = 4242)
    assert _walk(str(root)) == f"contains a nested host mount: {root / 'a' / 'b'}"
    assert os_sandbox.cache_share_hazard(str(root)) == (
        f"contains a nested host mount: {root / 'a' / 'b'}"
    )


def test_a_child_that_is_its_parent_again_is_a_nested_mount(tmp_path, monkeypatch):
    root = _tree(tmp_path / "root")
    parent = os.stat(root / "a")
    _Mounts(monkeypatch).mount(str(root / "a" / "b"), device = parent.st_dev, inode = parent.st_ino)
    assert _walk(str(root)) == f"contains a nested host mount: {root / 'a' / 'b'}"


def test_a_same_device_bind_is_walked_as_ismount_walks_it(tmp_path, monkeypatch):
    # ismount cannot see it either; the mount table check in front of the walk refuses it.
    root = _tree(tmp_path / "root")
    _Mounts(monkeypatch).mount(str(root / "c"), device = os.stat(root).st_dev, inode = 987654321)
    witness, expected = [], []
    assert _walk(str(root), witness = witness) is None
    assert _reference_walk(str(root), 50_000, expected) is None
    assert witness == expected


def test_a_mount_under_a_root_spelled_through_a_symlink_is_found(tmp_path, monkeypatch):
    root = _tree(tmp_path / "root")
    alias = tmp_path / "alias"
    alias.symlink_to(root)
    _Mounts(monkeypatch).mount(str(root / "c"), device = 4242)
    assert _walk(str(alias)) == f"contains a nested host mount: {alias / 'c'}"


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason = "root ignores the permission bits"
)
@pytest.mark.parametrize("mode", [0o000, 0o300, 0o500, 0o600])
def test_a_directory_closed_to_search_or_listing_answers_as_before(tmp_path, monkeypatch, mode):
    root = _tree(tmp_path / "root")
    closed = root / "a" / "b"
    (closed / "inner").mkdir()
    mounts = _Mounts(monkeypatch)
    mounts.mount(str(root / "c"), device = 4242)
    closed.chmod(mode)
    try:
        for top in (str(root), str(root / "a")):
            witness, expected = [], []
            assert _walk(top, witness = witness) == _reference_walk(top, 50_000, expected)
            assert witness == expected
    finally:
        closed.chmod(0o700)


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason = "root ignores the permission bits"
)
@pytest.mark.parametrize("seed", range(40))
def test_seeded_trees_answer_as_the_per_directory_ismount_walk(tmp_path, monkeypatch, seed):
    rnd = random.Random(seed)
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    dirs, files = [root], []
    for index in range(rnd.randint(4, 30)):
        parent = rnd.choice(dirs)
        path = parent / f"{rnd.choice(['d', 'nul', 'x y', '-n', 'é'])}{index}"
        if rnd.random() < 0.35:
            path.mkdir()
            dirs.append(path)
        else:
            path.write_text("x")
            files.append(path)
    mounts = _Mounts(monkeypatch)
    closed = []
    for index in range(rnd.randint(0, 3)):
        kind = rnd.choice(
            ["link_out", "link_in", "symlink", "fifo", "mount", "self", "bind", "closed"]
        )
        parent = rnd.choice(dirs)
        target = parent / f"feature{index}"
        if kind == "link_out":
            (outside / f"o{index}").write_text("o")
            os.link(outside / f"o{index}", target)
        elif kind == "link_in" and files:
            os.link(rnd.choice(files), target)
        elif kind == "symlink":
            target.symlink_to(rnd.choice([outside, root, parent]))
        elif kind == "fifo":
            os.mkfifo(target)
        elif kind in ("mount", "self", "bind") and len(dirs) > 1:
            child = rnd.choice(dirs[1:])
            above = os.stat(child.parent)
            if kind == "mount":
                mounts.mount(str(child), device = 4242 + index)
            elif kind == "self":
                mounts.mount(str(child), device = above.st_dev, inode = above.st_ino)
            else:
                mounts.mount(str(child), device = above.st_dev, inode = 900000000 + index)
        elif kind == "closed" and len(dirs) > 1:
            child = rnd.choice(dirs[1:])
            closed.append(child)
    alias = tmp_path / "alias"
    alias.symlink_to(root)
    # Deepest first: a closed parent would keep its child out of reach.
    for child in sorted(set(closed), key = lambda path: (-len(path.parts), str(path))):
        child.chmod(rnd.choice([0o000, 0o300, 0o600]))
    try:
        for top in (str(root), str(alias)):
            for budget in (50_000, rnd.randint(1, 12)):
                witness, expected = [], []
                assert _walk(top, budget, witness) == _reference_walk(top, budget, expected)
                assert witness == expected
                assert _walk(top, budget) == _reference_walk(top, budget, None)
    finally:
        # Parents first: a closed parent keeps its closed child out of reach.
        for child in sorted(set(closed), key = lambda path: (len(path.parts), str(path))):
            child.chmod(0o700)


def test_the_filesystem_root_finds_a_real_nested_mount():
    if not sys.platform.startswith("linux"):
        pytest.skip("reads the Linux mount layout")
    found = _walk("/", 1_000_000)
    assert found == _reference_walk("/", 1_000_000, None)
    if os.path.ismount("/proc"):
        assert found is not None and found.startswith("contains a nested host mount: /")

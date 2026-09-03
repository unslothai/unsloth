# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Restoring huggingface_hub's resumable HTTP partials.

The patched writer has to be a faithful 1.17 caller and nothing more: a stable name, opened for
append, told how far it got, and left alone on failure. It also has to stand down wherever it
cannot prove that is safe, which is the whole reason 1.18 removed it.
"""

from __future__ import annotations

from pathlib import Path
import builtins
import errno
import os
import sys
import types

import pytest

from hub.utils import resumable_partials as rp


@pytest.fixture(autouse = True)
def _fresh_probe():
    rp.reset_probe_cache_for_tests()
    yield
    rp.reset_probe_cache_for_tests()


def _fake_file_download(monkeypatch, *, xet_available = False):
    """A stand-in for huggingface_hub.file_download that records what the writer did."""
    calls = {"http_get": [], "stock": [], "moved": []}

    def http_get(
        url,
        handle,
        *,
        resume_size = 0,
        headers = None,
        expected_size = None,
        tqdm_class = None,
    ):
        calls["http_get"].append({"resume_size": resume_size, "mode": handle.mode})
        handle.write(b"x" * 10)

    def stock(**kwargs):
        calls["stock"].append(kwargs)

    module = types.ModuleType("huggingface_hub.file_download")
    module._download_to_tmp_and_move = stock
    module.http_get = http_get
    module._chmod_and_move = lambda src, dst: calls["moved"].append((src, dst))
    module._check_disk_space = lambda size, path: None
    module.is_xet_available = lambda: xet_available

    hub = types.ModuleType("huggingface_hub")
    hub.__version__ = "1.28.0"
    hub.file_download = module
    hub.constants = types.SimpleNamespace(HF_HUB_CACHE = None)

    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.file_download", module)
    monkeypatch.setattr(rp, "_exclusion_is_provable", lambda _c = None: True)
    return module, calls


def _patched_writer(module):
    return module._download_to_tmp_and_move


# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "version, expected",
    [("0.36.2", False), ("1.17.0", False), ("1.18.0", True), ("1.28.0", True), ("2.0.0", False)],
)
def test_only_the_versions_that_need_it_and_that_it_has_been_read_against(
    monkeypatch, version, expected
):
    module, _ = _fake_file_download(monkeypatch)
    sys.modules["huggingface_hub"].__version__ = version
    assert rp.can_restore_partials() is expected


def test_a_filesystem_that_grants_the_lock_twice_keeps_the_stock_writer(monkeypatch):
    _fake_file_download(monkeypatch)
    monkeypatch.setattr(rp, "_exclusion_is_provable", lambda _c = None: False)
    assert rp.can_restore_partials() is False
    assert rp.restore_resumable_partials() is False


def test_a_hub_missing_the_pieces_is_left_alone(monkeypatch):
    module, _ = _fake_file_download(monkeypatch)
    del module._chmod_and_move
    assert rp.can_restore_partials() is False


def test_the_lock_probe_reports_a_working_lock(tmp_path, monkeypatch):
    """The real probe, on the real filesystem the tests run on."""
    monkeypatch.setattr(rp, "_probe_dir", lambda _c = None: tmp_path)
    monkeypatch.setattr(rp, "_filesystem_is_local", lambda _d: True)
    assert rp._exclusion_is_provable() is True
    assert not list(tmp_path.iterdir()), "the probe left its file behind"


def test_the_probe_follows_the_cache_studio_is_using_now(tmp_path, monkeypatch):
    """Not the one this process booted with.

    huggingface_hub resolves HF_HUB_CACHE at import and moving the cache in Settings does not
    rewrite the live process, so probing the constant would judge a different filesystem than the
    one a freshly spawned worker writes its partial to.
    """
    live = tmp_path / "moved-cache"
    fake = types.ModuleType("utils.hf_cache_settings")
    fake.active_hf_hub_cache = lambda: str(live)
    monkeypatch.setitem(sys.modules, "utils.hf_cache_settings", fake)

    assert rp._probe_dir() == live
    assert live.is_dir(), "the probe did not create the cache root"


def test_moving_the_cache_re_probes_rather_than_reusing_the_old_verdict(tmp_path, monkeypatch):
    """Two roots can sit on filesystems that disagree about flock, so the verdict is per root."""
    probed: list[str] = []
    monkeypatch.setattr(rp, "_filesystem_is_local", lambda _d: True)
    monkeypatch.setattr(
        rp, "_lock_is_honoured_at", lambda directory: probed.append(directory) or True
    )

    first, second = tmp_path / "one", tmp_path / "two"
    monkeypatch.setattr(rp, "_probe_dir", lambda _c = None: first)
    rp._exclusion_is_provable()
    monkeypatch.setattr(rp, "_probe_dir", lambda _c = None: second)
    rp._exclusion_is_provable()

    assert probed == [str(first), str(second)]


def test_a_network_cache_keeps_the_stock_writer(tmp_path, monkeypatch):
    """A probe on this host cannot speak for another client.

    NFS mounted -o local_lock=flock keeps flock locks client-local, so two hosts each take the
    lock and neither is refused. Nothing measurable here would notice, so the mount type decides.
    """
    for fstype in ("nfs4", "lustre", "gpfs", "cifs"):
        rp.invalidate_probe_cache()
        monkeypatch.setattr(rp, "_mounts", lambda: [(str(tmp_path), fstype)], raising = False)
        assert rp._filesystem_is_local(str(tmp_path)) is False, fstype


def test_a_network_cache_stands_down_even_where_the_lock_probe_would_pass(tmp_path, monkeypatch):
    """Locality gates the probe, not the other way round: on NFS the probe passes and lies."""
    monkeypatch.setattr(rp, "_probe_dir", lambda _c = None: tmp_path)
    monkeypatch.setattr(rp, "_lock_is_honoured_at", lambda _d: True)
    monkeypatch.setattr(rp, "_filesystem_is_local", lambda _d: False)
    assert rp._exclusion_is_provable() is False


def test_an_unidentifiable_mount_is_not_treated_as_local(tmp_path, monkeypatch):
    """Failing closed: this decides whether to re-enable a shared writer."""
    rp.invalidate_probe_cache()
    monkeypatch.setattr(rp, "_mounts", lambda: [], raising = False)
    assert rp._filesystem_is_local(str(tmp_path)) is False


def test_a_local_disk_is_local(tmp_path):
    """The filesystem the tests actually run on."""
    rp.invalidate_probe_cache()
    assert rp._filesystem_is_local(str(tmp_path)) is True


def test_an_unrecognised_mount_type_is_not_treated_as_local(tmp_path, monkeypatch):
    """Locality is an allowlist, because a FUSE daemon picks its own name.

    Unless it negotiates FUSE_FLOCK_LOCKS the kernel answers flock locally (libfuse
    fuse_lowlevel.h), so a cache on object storage passes the same-host probe while excluding no
    other client. A blank fstype says nothing either. Neither may read as local.
    """
    for fstype in ("fuse.rclone", "fuse.s3fs", "fuse.gocryptfs", "somethingnew", ""):
        rp.invalidate_probe_cache()
        monkeypatch.setattr(rp, "_mounts", lambda: [(str(tmp_path), fstype)], raising = False)
        assert rp._filesystem_is_local(str(tmp_path)) is False, fstype


def test_the_known_local_filesystems_are_local(tmp_path, monkeypatch):
    """The allowlist still has to admit the disks people actually keep a cache on."""
    for fstype in ("ext4", "xfs", "btrfs", "zfs", "apfs", "NTFS", "tmpfs", "overlay"):
        rp.invalidate_probe_cache()
        monkeypatch.setattr(rp, "_mounts", lambda: [(str(tmp_path), fstype)], raising = False)
        assert rp._filesystem_is_local(str(tmp_path)) is True, fstype


def test_each_cache_root_is_judged_on_its_own_filesystem(tmp_path, monkeypatch):
    """Unsloth remembers several cache roots and they need not lock alike.

    One global verdict taken from the selected cache would have the boot sweep delete a local
    cache's still-appendable partials whenever the selected one sits on a network mount.
    """
    from hub.utils import hf_cache_state

    local_root, network_root = tmp_path / "local", tmp_path / "network"
    local_root.mkdir()
    network_root.mkdir()
    # Past the version gate, so the filesystem is what is left to decide.
    # And out through the public entry point, past the version gate.
    monkeypatch.setattr("huggingface_hub.__version__", "1.28.0", raising = False)
    monkeypatch.setattr(rp, "_hub_is_patchable", lambda: True)
    monkeypatch.setattr(
        rp,
        "_mounts",
        lambda: [(str(local_root), "ext4"), (str(network_root), "nfs4")],
        raising = False,
    )
    hf_cache_state.invalidate_partial_resumability()

    partial = f"{'a' * 40}.incomplete"
    assert hf_cache_state.partial_is_resumable(partial, local_root) is True
    assert hf_cache_state.partial_is_resumable(partial, network_root) is False
    hf_cache_state.invalidate_partial_resumability()


def test_a_named_root_that_is_gone_is_not_recreated(tmp_path):
    """Asking about a detached cache must not bring it back, nor claim it locks."""
    missing = tmp_path / "unplugged"
    assert rp._probe_dir(missing) is None
    assert not missing.exists()


def test_only_contention_counts_as_a_working_lock(tmp_path, monkeypatch):
    """A filesystem with no locking answers ENOLCK or EOPNOTSUPP.

    Reading either as "refused" would enable the shared writer on exactly the mounts that cannot
    support it, which is the opposite of what the probe is for. flock never answers EACCES; that
    is fcntl's.
    """
    import errno as errno_mod

    calls = {"n": 0}

    def flock(fd, op):
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        raise OSError(flock.errno_value, "nope")

    fake = types.ModuleType("fcntl")
    fake.flock = flock
    fake.LOCK_EX, fake.LOCK_NB = 2, 4
    monkeypatch.setitem(sys.modules, "fcntl", fake)

    for value, expected in (
        (errno_mod.EWOULDBLOCK, True),
        (errno_mod.EAGAIN, True),
        (errno_mod.ENOLCK, False),
        (errno_mod.EOPNOTSUPP, False),
        (errno_mod.EINTR, False),
        (errno_mod.EACCES, False),
    ):
        rp.invalidate_probe_cache()
        calls["n"] = 0
        flock.errno_value = value
        assert rp._lock_is_honoured_at(str(tmp_path)) is expected, errno_mod.errorcode[value]


def test_the_probe_does_not_follow_a_planted_symlink(tmp_path):
    """A shared cache is writable by others, and a predictable probe name is a truncation gadget.

    Plants a symlink at the name a pid-based scheme would pick, pointing at a file the Unsloth
    account owns. Opening that path "wb" follows the link and empties the target, so the probe has
    to use a name nobody can guess and create it exclusively.
    """
    import os as os_mod

    rp.invalidate_probe_cache()
    victim = tmp_path / "victim"
    victim.write_bytes(b"important")
    cache = tmp_path / "cache"
    cache.mkdir()
    planted = cache / (".unsloth-flock-probe.%d" % os_mod.getpid())
    planted.symlink_to(victim)

    assert rp._lock_is_honoured_at(str(cache)) is True
    assert victim.read_bytes() == b"important", "the probe followed the symlink and truncated it"
    assert planted.is_symlink(), "the probe wrote through the planted name"
    planted.unlink()
    assert not list(cache.iterdir()), "the probe left its own file behind"


def test_the_verdict_is_cached_per_directory(tmp_path):
    """The hot path asks per blob, so a repeat on the same root must not re-probe."""
    rp.invalidate_probe_cache()
    root = tmp_path / "cache"
    root.mkdir()
    assert rp._lock_is_honoured_at(str(root)) is True
    before = rp._lock_is_honoured_on.cache_info()
    assert rp._lock_is_honoured_at(str(root)) is True
    assert rp._lock_is_honoured_on.cache_info().hits == before.hits + 1


def test_a_new_mount_at_the_same_path_is_re_probed(tmp_path, monkeypatch):
    """The path is not the identity of a filesystem.

    An external cache can be unmounted and something else mounted at the same name, and a verdict
    about the old filesystem says nothing about the new one, so the device is part of the key.
    """
    rp.invalidate_probe_cache()
    root = tmp_path / "cache"
    root.mkdir()
    devices = iter([101, 101, 202])
    monkeypatch.setattr(rp, "_device_at", lambda _d: next(devices))

    assert rp._lock_is_honoured_at(str(root)) is True
    hits = rp._lock_is_honoured_on.cache_info().hits
    assert rp._lock_is_honoured_at(str(root)) is True
    assert rp._lock_is_honoured_on.cache_info().hits == hits + 1, "same device should have hit"
    misses = rp._lock_is_honoured_on.cache_info().misses
    assert rp._lock_is_honoured_at(str(root)) is True
    assert rp._lock_is_honoured_on.cache_info().misses == misses + 1, "remount was not re-probed"


def test_a_probe_that_could_not_run_is_not_remembered(tmp_path, monkeypatch):
    """A full disk or a briefly unwritable cache is not a measurement.

    Caching one would have the backend condemn partials a freshly spawned worker is happily
    resuming, for the life of the process.
    """
    rp.invalidate_probe_cache()
    attempts: list[int] = []

    def fail_once():
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("mount table briefly unreadable")
        return [(str(tmp_path), "ext4")]

    monkeypatch.setattr(rp, "_mounts", fail_once, raising = False)

    with pytest.raises(rp._ProbeUnavailable):
        rp._filesystem_is_local(str(tmp_path))
    assert rp._filesystem_is_local(str(tmp_path)) is True, "the failure was cached"


def test_an_unrunnable_probe_travels_up_rather_than_answering(tmp_path, monkeypatch):
    """The probe layer must not turn "could not tell" into "no", or a caller would cache it."""
    rp.invalidate_probe_cache()
    monkeypatch.setattr(rp, "_probe_dir", lambda _c = None: tmp_path)
    monkeypatch.setattr(
        rp, "_device_at", lambda _d: (_ for _ in ()).throw(rp._ProbeUnavailable("gone"))
    )
    with pytest.raises(rp._ProbeUnavailable):
        rp._exclusion_is_provable()

    monkeypatch.setattr("huggingface_hub.__version__", "1.28.0", raising = False)
    monkeypatch.setattr(rp, "_hub_is_patchable", lambda: True)
    with pytest.raises(rp._ProbeUnavailable):
        rp.can_restore_partials()


def test_the_capability_reads_false_when_the_probe_could_not_run(monkeypatch):
    """Fail closed at the boundary the callers use, and do not remember it there either."""
    from hub.utils import hf_cache_state

    monkeypatch.setattr("huggingface_hub.__version__", "1.28.0", raising = False)
    attempts: list[int] = []

    def unavailable(_c = None):
        attempts.append(1)
        raise rp._ProbeUnavailable("mount table briefly unreadable")

    monkeypatch.setattr(rp, "can_restore_partials", unavailable)
    hf_cache_state.invalidate_partial_resumability()

    assert hf_cache_state.hf_partials_are_resumable() is False
    assert hf_cache_state.hf_partials_are_resumable() is False
    assert len(attempts) == 2, "the failure was cached instead of being re-probed"


def test_changing_the_cache_home_invalidates_the_verdict(monkeypatch):
    """The one place the root can move at runtime must drop both cached answers."""
    from hub.utils import hf_cache_state

    monkeypatch.setattr(rp, "can_restore_partials", lambda _c = None: True)
    monkeypatch.setattr("huggingface_hub.__version__", "1.28.0", raising = False)
    hf_cache_state.invalidate_partial_resumability()
    assert hf_cache_state.hf_partials_are_resumable() is True

    monkeypatch.setattr(rp, "can_restore_partials", lambda _c = None: False)
    # No cache at this layer any more: the verdict follows the filesystem, and a result kept against the
    # path alone outlives a remount at the same name.
    assert hf_cache_state.hf_partials_are_resumable() is False
    hf_cache_state.invalidate_partial_resumability()


# ---------------------------------------------------------------------------------------------


def test_it_appends_to_the_stable_name_and_says_how_far_it_got(monkeypatch, tmp_path):
    module, calls = _fake_file_download(monkeypatch)
    assert rp.restore_resumable_partials() is True

    partial = tmp_path / "abc.incomplete"
    partial.write_bytes(b"y" * 40)
    destination = tmp_path / "abc"

    _patched_writer(module)(
        incomplete_path = partial,
        destination_path = destination,
        url_to_download = "https://example/f",
        headers = {},
        expected_size = 50,
        filename = "f",
    )

    assert calls["http_get"] == [{"resume_size": 40, "mode": "ab"}]
    assert calls["moved"] == [(partial, destination)]
    assert calls["stock"] == []


def test_a_planted_symlink_is_not_appended_to(monkeypatch, tmp_path):
    """The stable name is predictable, so on a shared cache it can be pre-created.

    An unguarded "ab" would follow the link and append the model to whatever it points at, and
    _chmod_and_move would then chmod that file too.
    """
    module, calls = _fake_file_download(monkeypatch)
    assert rp.restore_resumable_partials() is True

    victim = tmp_path / "victim"
    victim.write_bytes(b"keep me")
    partial = tmp_path / "abc.incomplete"
    partial.symlink_to(victim)

    _patched_writer(module)(
        incomplete_path = partial,
        destination_path = tmp_path / "abc",
        url_to_download = "https://example/f",
        headers = {},
        expected_size = 50,
        filename = "f",
    )

    assert victim.read_bytes() == b"keep me", "the download was appended to the symlink target"
    assert not partial.is_symlink(), "the planted link survived"
    # Started from zero rather than trusting the target's length.
    assert calls["http_get"] == [{"resume_size": 0, "mode": "ab"}]


def test_a_partial_left_by_another_user_is_not_built_on(monkeypatch, tmp_path):
    """Nothing about a plain file says who wrote it.

    A partial another account left is bytes of their choosing, and appending the server's
    remaining range to a chosen prefix publishes a blob of exactly the right length and entirely
    the wrong contents. huggingface_hub checks the size afterwards and never the hash
    (huggingface_hub#3643), so no later step would catch it.
    """
    module, calls = _fake_file_download(monkeypatch)
    assert rp.restore_resumable_partials() is True

    partial = tmp_path / "abc.incomplete"
    partial.write_bytes(b"poison" * 100)

    # Only the planted file reads as somebody else's, and the fresh partial replacing it has to still
    # be ours; keyed on which open it is, since a filesystem may reuse the released inode.
    real_fstat = os.fstat
    opens: list[int] = []

    class _Foreign:
        def __init__(self, info):
            self.st_mode, self.st_nlink = info.st_mode, info.st_nlink
            self.st_dev, self.st_ino = info.st_dev, info.st_ino
            self.st_uid = info.st_uid + 1

    def fstat(descriptor, *args, **kwargs):
        info = real_fstat(descriptor, *args, **kwargs)
        opens.append(1)
        return _Foreign(info) if len(opens) == 1 else info

    monkeypatch.setattr(rp.os, "fstat", fstat)

    _patched_writer(module)(
        incomplete_path = partial,
        destination_path = tmp_path / "abc",
        url_to_download = "https://example/f",
        headers = {},
        expected_size = 50,
        filename = "f",
    )

    assert calls["http_get"] == [{"resume_size": 0, "mode": "ab"}], "it resumed on foreign bytes"
    assert partial.read_bytes() == b"x" * 10, "the foreign prefix survived into the blob"


def test_an_unopenable_partial_defers_instead_of_failing_the_download(monkeypatch, tmp_path):
    """A 0600 partial from another account answers EACCES, and that is not a reason to give up.

    Stock writes a file of its own and never touches this one, so the download still happens.
    Raising here would fail every attempt at that blob for good, which is worse than not resuming.
    """
    module, calls = _fake_file_download(monkeypatch)
    assert rp.restore_resumable_partials() is True

    partial = tmp_path / "abc.incomplete"
    partial.write_bytes(b"someone else's")
    real_open = os.open

    def refuse(target, *args, **kwargs):
        if str(target) == str(partial):
            raise PermissionError(errno.EACCES, "Permission denied", str(target))
        return real_open(target, *args, **kwargs)

    monkeypatch.setattr(rp.os, "open", refuse)

    _patched_writer(module)(
        incomplete_path = partial,
        destination_path = tmp_path / "abc",
        url_to_download = "https://example/f",
        headers = {},
        expected_size = 50,
        filename = "f",
    )

    assert len(calls["stock"]) == 1, "the download was abandoned rather than handed to stock"
    assert calls["http_get"] == []
    assert partial.read_bytes() == b"someone else's", "it deleted a partial it could not read"


def test_a_partial_swapped_after_the_last_write_is_not_published(monkeypatch, tmp_path):
    """_chmod_and_move resolves the name again, so the name has to still hold what was written.

    Otherwise a shared cache lets another account replace the partial once writing stops and have
    its file installed as the blob, under a descriptor that passed every check.
    """
    module, calls = _fake_file_download(monkeypatch)
    assert rp.restore_resumable_partials() is True

    partial = tmp_path / "abc.incomplete"
    real_http_get = module.http_get

    def swap_then_write(url, handle, **kwargs):
        real_http_get(url, handle, **kwargs)
        # The writer is finished with the descriptor; the name now points somewhere else.
        partial.unlink()
        partial.write_bytes(b"attacker's model")

    module.http_get = swap_then_write

    _patched_writer(module)(
        incomplete_path = partial,
        destination_path = tmp_path / "abc",
        url_to_download = "https://example/f",
        headers = {},
        expected_size = 50,
        filename = "f",
    )

    assert calls["moved"] == [], "the replacement was published as the blob"
    assert partial.read_bytes() == b"attacker's model", "it should be left for the retry to judge"


def test_ownership_that_cannot_be_established_is_an_objection(monkeypatch, tmp_path):
    """Windows has no st_uid and no ACL read without pywin32, so nothing there can be vouched for.

    _exclusion_is_provable keeps the shared writer off that platform, and this is the second line:
    if it were ever enabled, an unknown owner still must not read as ours.
    """
    monkeypatch.delattr(rp.os, "geteuid", raising = False)
    target = tmp_path / "partial"
    target.write_bytes(b"whoever wrote this")
    descriptor = os.open(target, os.O_RDONLY)
    try:
        assert rp._objection_to(descriptor) is not None
    finally:
        os.close(descriptor)


def test_windows_keeps_the_stock_writer(monkeypatch, tmp_path):
    """No fcntl and no way to establish ownership, so the shared name stays off.

    os.name is faked as well as fcntl: on a posix box the answer would be False either way, and
    a test that cannot tell the difference would not notice the writer being switched back on.
    """
    monkeypatch.setattr(rp, "_probe_dir", lambda _c = None: tmp_path)
    monkeypatch.setattr(rp, "_filesystem_is_local", lambda _d: True)
    monkeypatch.setattr(rp.os, "name", "nt")
    real_import = builtins.__import__

    def without_fcntl(name, *args, **kwargs):
        if name == "fcntl":
            raise ImportError("no fcntl on Windows")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_fcntl)
    assert rp._exclusion_is_provable() is False


def test_an_oversized_partial_is_restarted_rather_than_resumed(monkeypatch, tmp_path):
    """A partial longer than the file cannot be resumed from.

    A Range starting past the end answers 416, and it would answer 416 on every later attempt too,
    so the download would be wedged where the stock writer's fresh name recovers.
    """
    module, calls = _fake_file_download(monkeypatch)
    assert rp.restore_resumable_partials() is True

    partial = tmp_path / "abc.incomplete"
    partial.write_bytes(b"z" * 80)

    _patched_writer(module)(
        incomplete_path = partial,
        destination_path = tmp_path / "abc",
        url_to_download = "https://example/f",
        headers = {},
        expected_size = 50,
        filename = "f",
    )

    assert calls["http_get"] == [{"resume_size": 0, "mode": "ab"}]
    assert partial.read_bytes() == b"x" * 10, "the oversized bytes were kept"


def test_a_complete_partial_is_left_for_upstream_to_finish(monkeypatch, tmp_path):
    """Exactly the declared size is not the wedged case: http_get returns early and it publishes.

    Restarting here would refetch the whole file for nothing, so the boundary is strictly greater.
    """
    module, calls = _fake_file_download(monkeypatch)
    assert rp.restore_resumable_partials() is True

    partial = tmp_path / "abc.incomplete"
    partial.write_bytes(b"z" * 50)

    _patched_writer(module)(
        incomplete_path = partial,
        destination_path = tmp_path / "abc",
        url_to_download = "https://example/f",
        headers = {},
        expected_size = 50,
        filename = "f",
    )

    assert calls["http_get"] == [{"resume_size": 50, "mode": "ab"}], "it restarted a finished file"


def test_an_unavailable_probe_does_not_take_the_worker_down(monkeypatch):
    """The worker patches at import, so this must never raise out of here.

    _ProbeUnavailable is what a transient mount-table or write-probe failure raises; escaping it
    would kill the download process instead of leaving it with the stock writer.
    """
    _fake_file_download(monkeypatch)

    def unavailable(_c = None):
        raise rp._ProbeUnavailable("mount table briefly unreadable")

    monkeypatch.setattr(rp, "can_restore_partials", unavailable)
    assert rp.restore_resumable_partials() is False


def test_a_planted_hard_link_is_not_appended_to(monkeypatch, tmp_path):
    """O_NOFOLLOW cannot see a hard link, so the size check is what catches this one."""
    module, calls = _fake_file_download(monkeypatch)
    assert rp.restore_resumable_partials() is True

    victim = tmp_path / "victim"
    victim.write_bytes(b"keep me")
    partial = tmp_path / "abc.incomplete"
    os.link(victim, partial)

    _patched_writer(module)(
        incomplete_path = partial,
        destination_path = tmp_path / "abc",
        url_to_download = "https://example/f",
        headers = {},
        expected_size = 50,
        filename = "f",
    )

    assert victim.read_bytes() == b"keep me", "the download was appended through the hard link"
    assert calls["http_get"] == [{"resume_size": 0, "mode": "ab"}]


def test_a_planted_partial_that_cannot_be_removed_defers_to_stock(monkeypatch, tmp_path):
    """Stock invents its own name, so it cannot be steered by a planted one either."""
    module, calls = _fake_file_download(monkeypatch)
    assert rp.restore_resumable_partials() is True

    victim = tmp_path / "victim"
    victim.write_bytes(b"keep me")
    partial = tmp_path / "abc.incomplete"
    partial.symlink_to(victim)

    def refuse(_path):
        raise PermissionError("sticky directory")

    monkeypatch.setattr(rp.os, "unlink", refuse)

    _patched_writer(module)(
        incomplete_path = partial,
        destination_path = tmp_path / "abc",
        url_to_download = "https://example/f",
        headers = {},
        expected_size = 50,
        filename = "f",
    )

    assert calls["http_get"] == [], "the patched writer wrote anyway"
    assert len(calls["stock"]) == 1
    assert victim.read_bytes() == b"keep me"


def test_a_failed_download_leaves_the_partial_for_the_next_attempt(monkeypatch, tmp_path):
    module, _ = _fake_file_download(monkeypatch)
    rp.restore_resumable_partials()

    def boom(*_args, **_kwargs):
        raise OSError("connection reset")

    module.http_get = boom
    partial = tmp_path / "abc.incomplete"
    partial.write_bytes(b"y" * 40)

    with pytest.raises(OSError):
        _patched_writer(module)(
            incomplete_path = partial,
            destination_path = tmp_path / "abc",
            url_to_download = "https://example/f",
            headers = {},
            expected_size = 50,
            filename = "f",
        )
    assert partial.exists() and partial.stat().st_size == 40


def test_xet_keeps_its_own_writer(monkeypatch, tmp_path):
    module, calls = _fake_file_download(monkeypatch, xet_available = True)
    rp.restore_resumable_partials()

    _patched_writer(module)(
        incomplete_path = tmp_path / "abc.incomplete",
        destination_path = tmp_path / "abc",
        url_to_download = "https://example/f",
        headers = {},
        expected_size = 50,
        filename = "f",
        xet_file_data = object(),
    )
    assert calls["http_get"] == [] and len(calls["stock"]) == 1


def test_a_xet_backed_repo_downloading_over_http_still_resumes(monkeypatch, tmp_path):
    """xet_file_data is set for any XET-backed repo, including when hf_xet is off.

    Gating on the metadata rather than on whether XET will actually run silently disables this for
    most of the Hub.
    """
    module, calls = _fake_file_download(monkeypatch, xet_available = False)
    rp.restore_resumable_partials()

    _patched_writer(module)(
        incomplete_path = tmp_path / "abc.incomplete",
        destination_path = tmp_path / "abc",
        url_to_download = "https://example/f",
        headers = {},
        expected_size = 50,
        filename = "f",
        xet_file_data = object(),
    )
    assert len(calls["http_get"]) == 1 and calls["stock"] == []


def test_force_download_defers_to_stock(monkeypatch, tmp_path):
    module, calls = _fake_file_download(monkeypatch)
    rp.restore_resumable_partials()

    _patched_writer(module)(
        incomplete_path = tmp_path / "abc.incomplete",
        destination_path = tmp_path / "abc",
        url_to_download = "https://example/f",
        headers = {},
        expected_size = 50,
        filename = "f",
        force_download = True,
    )
    assert calls["http_get"] == [] and len(calls["stock"]) == 1


def test_an_already_downloaded_blob_is_not_fetched_again(monkeypatch, tmp_path):
    module, calls = _fake_file_download(monkeypatch)
    rp.restore_resumable_partials()

    destination = tmp_path / "abc"
    destination.write_bytes(b"done")
    _patched_writer(module)(
        incomplete_path = tmp_path / "abc.incomplete",
        destination_path = destination,
        url_to_download = "https://example/f",
        headers = {},
        expected_size = 50,
        filename = "f",
    )
    assert calls["http_get"] == [] and calls["stock"] == []


def test_patching_twice_keeps_one_layer(monkeypatch):
    module, _ = _fake_file_download(monkeypatch)
    assert rp.restore_resumable_partials() is True
    first = module._download_to_tmp_and_move
    assert rp.restore_resumable_partials() is True
    assert module._download_to_tmp_and_move is first


# ---------------------------------------------------------------------------------------------


def test_the_ui_is_told_partials_are_resumable_again(monkeypatch):
    from hub.utils import hf_cache_state

    _fake_file_download(monkeypatch)
    hf_cache_state.invalidate_partial_resumability()
    monkeypatch.setattr(rp, "can_restore_partials", lambda _c = None: True)
    assert hf_cache_state.hf_partials_are_resumable() is True

    hf_cache_state.invalidate_partial_resumability()
    monkeypatch.setattr(rp, "can_restore_partials", lambda _c = None: False)
    assert hf_cache_state.hf_partials_are_resumable() is False
    hf_cache_state.invalidate_partial_resumability()


def test_the_worker_restores_it_on_import():
    """A structural check: moving the call out of the worker fails here, not in the field."""
    import ast

    source = (Path(rp.__file__).parent.parent / "workers" / "hf_download.py").read_text(
        encoding = "utf-8"
    )
    tree = ast.parse(source)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "restore_resumable_partials" in calls

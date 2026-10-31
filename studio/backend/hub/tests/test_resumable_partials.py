# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The patched writer must match the 1.17 caller and stand down whenever safety cannot be proven."""

from __future__ import annotations

from pathlib import Path
import builtins
import errno
import os
import sys
import types

import pytest

from hub.utils import resumable_partials as rp


def _shared_setup_1(module, partial, tmp_path):
    _patched_writer(module)(
        incomplete_path = partial,
        destination_path = tmp_path / "abc",
        url_to_download = "https://example/f",
        headers = {},
        expected_size = 50,
        filename = "f",
    )


def _shared_setup_2(monkeypatch, tmp_path):
    module, calls = _fake_file_download(monkeypatch)
    assert rp.restore_resumable_partials() is True

    partial = tmp_path / "abc.incomplete"
    return calls, module, partial


def _shared_setup_3(monkeypatch, tmp_path):
    module, calls = _fake_file_download(monkeypatch)
    assert rp.restore_resumable_partials() is True

    victim = tmp_path / "victim"
    victim.write_bytes(b"keep me")
    partial = tmp_path / "abc.incomplete"
    return calls, module, partial, victim


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
    """Probe the cache Studio uses now, not HF_HUB_CACHE, which huggingface_hub resolved at import."""
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
    """Network mounts are non-local, since NFS local_lock=flock keeps flock locks client-local."""
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
    """Locality is an allowlist: a FUSE mount that lacks FUSE_FLOCK_LOCKS answers flock locally."""
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
    """Each cache root is judged on its own filesystem; a network root must not condemn local partials."""
    from hub.utils import hf_cache_state

    local_root, network_root = tmp_path / "local", tmp_path / "network"
    local_root.mkdir()
    network_root.mkdir()
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
    """ENOLCK and EOPNOTSUPP mean no locking, so only lock contention may count as a working lock."""
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
    """Probe files need unguessable names created exclusively, or a planted symlink truncates its target."""
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
    """Probe verdicts are keyed by device, since a new mount at the same path is a new filesystem."""
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
    """A probe that could not run must not be cached, or a full disk condemns partials until restart."""
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
    assert hf_cache_state.hf_partials_are_resumable() is False
    hf_cache_state.invalidate_partial_resumability()


def test_it_appends_to_the_stable_name_and_says_how_far_it_got(monkeypatch, tmp_path):
    calls, module, partial = _shared_setup_2(monkeypatch, tmp_path)
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
    """An append-mode open on the stable name would follow a planted link into the victim file."""
    calls, module, partial, victim = _shared_setup_3(monkeypatch, tmp_path)
    partial.symlink_to(victim)

    _shared_setup_1(module, partial, tmp_path)

    assert victim.read_bytes() == b"keep me", "the download was appended to the symlink target"
    assert not partial.is_symlink(), "the planted link survived"
    assert calls["http_get"] == [{"resume_size": 0, "mode": "ab"}]


def test_a_partial_left_by_another_user_is_not_built_on(monkeypatch, tmp_path):
    """Never append to a partial another account wrote; upstream checks size, never the hash."""
    calls, module, partial = _shared_setup_2(monkeypatch, tmp_path)
    partial.write_bytes(b"poison" * 100)

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

    _shared_setup_1(module, partial, tmp_path)

    assert calls["http_get"] == [{"resume_size": 0, "mode": "ab"}], "it resumed on foreign bytes"
    assert partial.read_bytes() == b"x" * 10, "the foreign prefix survived into the blob"


def test_an_unopenable_partial_defers_instead_of_failing_the_download(monkeypatch, tmp_path):
    """EACCES on another account's partial must defer to the stock writer, not fail the download for
    good."""
    calls, module, partial = _shared_setup_2(monkeypatch, tmp_path)
    partial.write_bytes(b"someone else's")
    real_open = os.open

    def refuse(target, *args, **kwargs):
        if str(target) == str(partial):
            raise PermissionError(errno.EACCES, "Permission denied", str(target))
        return real_open(target, *args, **kwargs)

    monkeypatch.setattr(rp.os, "open", refuse)

    _shared_setup_1(module, partial, tmp_path)

    assert len(calls["stock"]) == 1, "the download was abandoned rather than handed to stock"
    assert calls["http_get"] == []
    assert partial.read_bytes() == b"someone else's", "it deleted a partial it could not read"


def test_a_partial_swapped_after_the_last_write_is_not_published(monkeypatch, tmp_path):
    """_chmod_and_move re-resolves the name, so the partial must still be the file written, not a swap."""
    calls, module, partial = _shared_setup_2(monkeypatch, tmp_path)
    real_http_get = module.http_get

    def swap_then_write(url, handle, **kwargs):
        real_http_get(url, handle, **kwargs)
        partial.unlink()
        partial.write_bytes(b"attacker's model")

    module.http_get = swap_then_write

    _shared_setup_1(module, partial, tmp_path)

    assert calls["moved"] == [], "the replacement was published as the blob"
    assert partial.read_bytes() == b"attacker's model", "it should be left for the retry to judge"


def test_ownership_that_cannot_be_established_is_an_objection(monkeypatch, tmp_path):
    """An owner that cannot be established is an objection, so an unknown owner never reads as ours."""
    monkeypatch.delattr(rp.os, "geteuid", raising = False)
    target = tmp_path / "partial"
    target.write_bytes(b"whoever wrote this")
    descriptor = os.open(target, os.O_RDONLY)
    try:
        assert rp._objection_to(descriptor) is not None
    finally:
        os.close(descriptor)


def test_windows_keeps_the_stock_writer(monkeypatch, tmp_path):
    """Without fcntl or provable ownership the shared name stays off; os.name and fcntl are both faked."""
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
    """An oversized partial must restart, since a Range past its end answers 416 on every attempt."""
    calls, module, partial = _shared_setup_2(monkeypatch, tmp_path)
    partial.write_bytes(b"z" * 80)

    _shared_setup_1(module, partial, tmp_path)

    assert calls["http_get"] == [{"resume_size": 0, "mode": "ab"}]
    assert partial.read_bytes() == b"x" * 10, "the oversized bytes were kept"


def test_a_complete_partial_is_left_for_upstream_to_finish(monkeypatch, tmp_path):
    """A partial of exactly the declared size is left alone; restart only when it is strictly larger."""
    calls, module, partial = _shared_setup_2(monkeypatch, tmp_path)
    partial.write_bytes(b"z" * 50)

    _shared_setup_1(module, partial, tmp_path)

    assert calls["http_get"] == [{"resume_size": 50, "mode": "ab"}], "it restarted a finished file"


def test_an_unavailable_probe_does_not_take_the_worker_down(monkeypatch):
    """A transient _ProbeUnavailable must not escape the import-time patch and kill the download worker."""
    _fake_file_download(monkeypatch)

    def unavailable(_c = None):
        raise rp._ProbeUnavailable("mount table briefly unreadable")

    monkeypatch.setattr(rp, "can_restore_partials", unavailable)
    assert rp.restore_resumable_partials() is False


def test_a_planted_hard_link_is_not_appended_to(monkeypatch, tmp_path):
    """O_NOFOLLOW cannot see a hard link, so the size check is what catches this one."""
    calls, module, partial, victim = _shared_setup_3(monkeypatch, tmp_path)
    os.link(victim, partial)

    _shared_setup_1(module, partial, tmp_path)

    assert victim.read_bytes() == b"keep me", "the download was appended through the hard link"
    assert calls["http_get"] == [{"resume_size": 0, "mode": "ab"}]


def test_a_planted_partial_that_cannot_be_removed_defers_to_stock(monkeypatch, tmp_path):
    """Stock invents its own name, so it cannot be steered by a planted one either."""
    calls, module, partial, victim = _shared_setup_3(monkeypatch, tmp_path)
    partial.symlink_to(victim)

    def refuse(_path):
        raise PermissionError("sticky directory")

    monkeypatch.setattr(rp.os, "unlink", refuse)

    _shared_setup_1(module, partial, tmp_path)

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
    """Gate on whether XET will actually run, not on xet_file_data, which is set for every XET repo."""
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

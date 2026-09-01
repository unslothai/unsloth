# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import importlib.util
import os
import socket
from pathlib import Path

import pytest


def _load_sandbox_module():
    path = Path(__file__).resolve().parents[1] / "core" / "inference" / "sandbox.py"
    spec = importlib.util.spec_from_file_location("_studio_sandbox_profile_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_python_read_paths_rejects_filesystem_roots(monkeypatch):
    sandbox = _load_sandbox_module()
    root = os.path.abspath(os.sep)

    monkeypatch.setattr(sandbox.sys, "prefix", root)
    monkeypatch.setattr(sandbox.sys, "base_prefix", root)
    monkeypatch.setattr(sandbox.site, "getsitepackages", lambda: [root])
    monkeypatch.setattr(sandbox, "_editable_source_paths", lambda: [root])

    assert root not in sandbox._python_read_paths()


def test_python_read_paths_rejects_home_and_ancestors_but_keeps_nested_runtime(
    tmp_path, monkeypatch
):
    sandbox = _load_sandbox_module()
    home = tmp_path / "home" / "user"
    nested_runtime = home / "project" / ".venv"
    nested_runtime.mkdir(parents = True)

    monkeypatch.setattr(sandbox.os.path, "expanduser", lambda _path: str(home))
    monkeypatch.setattr(sandbox.sys, "prefix", str(home))
    monkeypatch.setattr(sandbox.sys, "base_prefix", str(tmp_path))
    monkeypatch.setattr(sandbox.site, "getsitepackages", lambda: [str(nested_runtime)])
    monkeypatch.setattr(sandbox, "_editable_source_paths", lambda: [])

    paths = sandbox._python_read_paths()
    assert os.path.realpath(home) not in paths
    assert os.path.realpath(tmp_path) not in paths
    assert os.path.realpath(nested_runtime) in paths


def test_python_read_paths_narrows_shared_home_local_runtime(tmp_path, monkeypatch):
    sandbox = _load_sandbox_module()
    home = tmp_path / "home"
    shared_local = home / ".local"
    runtime_dirs = [shared_local / name for name in ("bin", "lib", "lib64")]
    for path in runtime_dirs:
        path.mkdir(parents = True, exist_ok = True)
    (shared_local / "share" / "private-app").mkdir(parents = True)
    (shared_local / "state" / "private-app").mkdir(parents = True)

    monkeypatch.setattr(sandbox.os.path, "expanduser", lambda _path: str(home))
    monkeypatch.setattr(sandbox.sys, "prefix", str(shared_local))
    monkeypatch.setattr(sandbox.sys, "base_prefix", str(shared_local))
    monkeypatch.setattr(sandbox.site, "getsitepackages", lambda: [])
    monkeypatch.setattr(sandbox, "_editable_source_paths", lambda: [])

    paths = sandbox._python_read_paths()
    assert os.path.realpath(shared_local) not in paths
    assert all(os.path.realpath(path) in paths for path in runtime_dirs)
    assert os.path.realpath(shared_local / "share") not in paths
    assert os.path.realpath(shared_local / "state") not in paths


def test_python_read_paths_includes_source_tree_sandbox_site(tmp_path, monkeypatch):
    sandbox = _load_sandbox_module()
    prefix = tmp_path / "python"
    prefix.mkdir()

    monkeypatch.setattr(sandbox.sys, "prefix", str(prefix))
    monkeypatch.setattr(sandbox.sys, "base_prefix", str(prefix))
    monkeypatch.setattr(sandbox.site, "getsitepackages", lambda: [])
    monkeypatch.setattr(sandbox, "_editable_source_paths", lambda: [])

    assert os.path.realpath(sandbox._SANDBOX_SITE_DIR) in sandbox._python_read_paths()


def test_python_read_paths_includes_nix_store_for_nix_runtime(tmp_path, monkeypatch):
    sandbox = _load_sandbox_module()
    nix_store = tmp_path / "nix" / "store"
    prefix = nix_store / "hash-python"
    prefix.mkdir(parents = True)

    monkeypatch.setattr(sandbox, "_NIX_STORE", str(nix_store))
    monkeypatch.setattr(sandbox.sys, "prefix", str(prefix))
    monkeypatch.setattr(sandbox.sys, "base_prefix", str(prefix))
    monkeypatch.setattr(sandbox.site, "getsitepackages", lambda: [])
    monkeypatch.setattr(sandbox, "_editable_source_paths", lambda: [])

    assert os.path.realpath(nix_store) in sandbox._python_read_paths()


def test_linux_ca_mounts_exclude_private_key_directories(tmp_path, monkeypatch):
    sandbox = _load_sandbox_module()
    monkeypatch.setattr(sandbox, "_linux_bwrap_path", "/usr/bin/bwrap")
    monkeypatch.setattr(sandbox, "_python_read_paths", lambda: [])

    argv = sandbox._linux_bwrap_argv(["/usr/bin/true"], str(tmp_path))
    targets = {
        argv[index + 2]
        for index, token in enumerate(argv)
        if token in {"--ro-bind", "--ro-bind-try", "--bind", "--bind-try"} and index + 2 < len(argv)
    }

    assert "/etc/ssl" not in targets
    assert "/etc/pki" not in targets
    assert "/etc/ssl/private" not in targets
    assert "/etc/pki/tls/private" not in targets
    assert "/etc/ssl/certs" in targets
    assert "/etc/pki/tls/certs" in targets


def test_linux_restores_accelerator_devices_after_synthetic_dev(tmp_path, monkeypatch):
    sandbox = _load_sandbox_module()
    monkeypatch.setattr(sandbox, "_linux_bwrap_path", "/usr/bin/bwrap")
    monkeypatch.setattr(sandbox, "_python_read_paths", lambda: [])

    argv = sandbox._linux_bwrap_argv(["/usr/bin/true"], str(tmp_path))
    device_targets = {
        argv[index + 2]
        for index, token in enumerate(argv)
        if token == "--dev-bind-try" and index + 2 < len(argv)
    }

    assert "/dev/dxg" in device_targets
    assert "/dev/dri" in device_targets
    assert "/dev/kfd" in device_targets
    assert "/dev/nvidiactl" in device_targets
    assert "/dev/nvidia-uvm" in device_targets


def test_linux_restores_accelerator_sysfs_class_and_backing_tree(tmp_path, monkeypatch):
    sandbox = _load_sandbox_module()
    sysfs_root = tmp_path / "sys"
    class_root = sysfs_root / "class" / "drm"
    device = sysfs_root / "devices" / "pci0000" / "card0"
    backing = device / "drm" / "card0"
    class_root.mkdir(parents = True)
    backing.mkdir(parents = True)
    try:
        os.symlink(backing, class_root / "card0", target_is_directory = True)
        os.symlink(device, backing / "device", target_is_directory = True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    monkeypatch.setattr(sandbox, "_LINUX_ACCELERATOR_SYSFS_CLASS_PATHS", (str(class_root),))
    assert sandbox._linux_accelerator_sysfs_paths() == [
        str(class_root),
        os.path.realpath(backing),
        os.path.realpath(device),
    ]


def test_linux_argv_mounts_private_shm_and_synthetic_identity(tmp_path, monkeypatch):
    sandbox = _load_sandbox_module()
    passwd_path = tmp_path / "passwd"
    group_path = tmp_path / "group"
    passwd_path.write_text("sandbox:x:1:1:Sandbox User:/tmp:/bin/sh\n")
    group_path.write_text("sandbox:x:1:\n")
    monkeypatch.setattr(sandbox, "_linux_bwrap_path", "/usr/bin/bwrap")
    monkeypatch.setattr(sandbox, "_python_read_paths", lambda: [])
    monkeypatch.setattr(
        sandbox, "_linux_sandbox_identity_files", lambda: (str(passwd_path), str(group_path))
    )

    argv = sandbox._linux_bwrap_argv(["/usr/bin/true"], str(tmp_path))
    assert ["--tmpfs", "/dev/shm"] == argv[argv.index("/dev/shm") - 1 : argv.index("/dev/shm") + 1]
    assert ["--ro-bind", str(passwd_path), "/etc/passwd"] == argv[
        argv.index("/etc/passwd") - 2 : argv.index("/etc/passwd") + 1
    ]
    assert ["--ro-bind", str(group_path), "/etc/group"] == argv[
        argv.index("/etc/group") - 2 : argv.index("/etc/group") + 1
    ]


def test_external_workdir_hardlink_is_rejected(tmp_path):
    sandbox = _load_sandbox_module()
    outside = tmp_path / "outside.txt"
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    outside.write_text("host data")
    try:
        os.link(outside, workdir / "linked.txt")
    except OSError as exc:
        pytest.skip(f"hard links unavailable on this test filesystem: {exc}")

    with pytest.raises(sandbox.UnsafeSandboxWorkdirError, match = "hard-linked outside"):
        sandbox._assert_no_external_hardlinks(str(workdir))


def test_internal_only_workdir_hardlinks_remain_allowed(tmp_path):
    sandbox = _load_sandbox_module()
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    original = workdir / "original.txt"
    original.write_text("session data")
    try:
        os.link(original, workdir / "alias.txt")
    except OSError as exc:
        pytest.skip(f"hard links unavailable on this test filesystem: {exc}")

    sandbox._assert_no_external_hardlinks(str(workdir))


def test_workdir_scan_fails_closed_above_entry_budget(tmp_path, monkeypatch):
    sandbox = _load_sandbox_module()
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    for index in range(3):
        (workdir / f"{index}.txt").write_text("x")
    monkeypatch.setattr(sandbox, "_WORKDIR_SCAN_MAX_ENTRIES", 2)

    with pytest.raises(sandbox.UnsafeSandboxWorkdirError, match = "entry limit"):
        sandbox._assert_no_external_hardlinks(str(workdir))


def test_workdir_scan_fails_closed_above_depth_budget(tmp_path, monkeypatch):
    sandbox = _load_sandbox_module()
    workdir = tmp_path / "workdir"
    (workdir / "a" / "b").mkdir(parents = True)
    monkeypatch.setattr(sandbox, "_WORKDIR_SCAN_MAX_DEPTH", 1)

    with pytest.raises(sandbox.UnsafeSandboxWorkdirError, match = "depth limit"):
        sandbox._assert_no_external_hardlinks(str(workdir))


def test_linux_mountinfo_detects_same_filesystem_bind_mount(tmp_path, monkeypatch):
    sandbox = _load_sandbox_module()
    workdir = tmp_path / "work dir"
    mounted = workdir / "mounted"
    mounted.mkdir(parents = True)
    mountinfo = tmp_path / "mountinfo"
    escaped_workdir = str(workdir).replace(" ", "\\040")
    escaped_mounted = str(mounted).replace(" ", "\\040")
    mountinfo.write_text(
        f"1 0 8:1 / {escaped_workdir} rw - ext4 /dev/sda1 rw\n"
        f"2 1 8:1 /host/data {escaped_mounted} rw - ext4 /dev/sda1 rw,bind\n"
    )
    monkeypatch.setattr(sandbox.sys, "platform", "linux")
    monkeypatch.setattr(sandbox, "_LINUX_MOUNTINFO", str(mountinfo))

    assert sandbox._linux_nested_mount_points(str(workdir)) == {os.path.normpath(mounted)}
    with pytest.raises(sandbox.UnsafeSandboxWorkdirError, match = "nested mount point"):
        sandbox._assert_no_external_hardlinks(str(workdir))


def test_workdir_socket_is_rejected(tmp_path):
    sandbox = _load_sandbox_module()
    if not hasattr(socket, "AF_UNIX"):
        pytest.skip("AF_UNIX unavailable")
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.bind(str(workdir / "service.sock"))
    except OSError as exc:
        sock.close()
        pytest.skip(f"pathname Unix sockets unavailable: {exc}")
    try:
        with pytest.raises(sandbox.UnsafeSandboxWorkdirError, match = "special filesystem node"):
            sandbox._assert_no_external_hardlinks(str(workdir))
    finally:
        sock.close()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason = "FIFOs unavailable")
def test_workdir_fifo_is_rejected(tmp_path):
    sandbox = _load_sandbox_module()
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    os.mkfifo(workdir / "host.fifo")

    with pytest.raises(sandbox.UnsafeSandboxWorkdirError, match = "special filesystem node"):
        sandbox._assert_no_external_hardlinks(str(workdir))


@pytest.mark.skipif(os.name == "nt", reason = "POSIX executable symlink layout")
def test_linux_recreates_unbound_executable_symlink(tmp_path, monkeypatch):
    sandbox = _load_sandbox_module()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    target = runtime / "python-real"
    target.write_text("binary")
    launcher = runtime / "python"
    try:
        os.symlink("python-real", launcher)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.setattr(sandbox, "_linux_bwrap_path", "/usr/bin/bwrap")
    monkeypatch.setattr(sandbox, "_python_read_paths", lambda: [])
    monkeypatch.setattr(sandbox.sys, "executable", str(launcher))
    argv = sandbox._linux_bwrap_argv(["/usr/bin/true"], str(workdir))

    link_index = argv.index("--symlink")
    assert argv[link_index + 1 : link_index + 3] == ["python-real", str(launcher)]
    assert not any(
        argv[index : index + 3] == ["--ro-bind-try", str(launcher), str(launcher)]
        for index in range(len(argv) - 2)
    )


def test_linux_reapplies_nested_runtime_after_writable_workdir(tmp_path, monkeypatch):
    sandbox = _load_sandbox_module()
    workdir = tmp_path / "project"
    runtime = workdir / ".venv"
    runtime.mkdir(parents = True)
    wd = os.path.realpath(workdir)
    rp = os.path.realpath(runtime)

    monkeypatch.setattr(sandbox, "_linux_bwrap_path", "/usr/bin/bwrap")
    monkeypatch.setattr(sandbox, "_python_read_paths", lambda: [rp])
    argv = sandbox._linux_bwrap_argv(["/usr/bin/true"], str(workdir))

    writable_index = next(
        index
        for index, token in enumerate(argv)
        if token == "--bind" and argv[index + 1 : index + 3] == [wd, wd]
    )
    readonly_indices = [
        index
        for index, token in enumerate(argv)
        if token == "--ro-bind-try" and argv[index + 1 : index + 3] == [rp, rp]
    ]
    assert readonly_indices[-1] > writable_index


def test_macos_profile_allows_child_signals_and_dev_null_writes(monkeypatch):
    sandbox = _load_sandbox_module()
    monkeypatch.setattr(sandbox, "_python_read_paths", lambda: [])
    monkeypatch.setattr(sandbox, "_assert_no_external_hardlinks", lambda _path: None)
    monkeypatch.setattr(sandbox.os.path, "realpath", lambda path: path)
    profile = sandbox._macos_seatbelt_profile("/tmp/work")

    assert "(allow signal (target same-sandbox))" in profile
    assert "(allow signal (target self))" not in profile
    assert "(allow file-write-data" in profile
    assert '(path "/dev/null")' in profile
    assert "(vnode-type CHARACTER-DEVICE)" in profile
    assert '(subpath "/dev/fd")' in profile


def test_macos_profile_allows_installed_developer_toolchain(monkeypatch):
    sandbox = _load_sandbox_module()
    developer = "/Library/Developer/CommandLineTools"
    monkeypatch.setattr(sandbox, "_python_read_paths", lambda: [])
    monkeypatch.setattr(sandbox, "_assert_no_external_hardlinks", lambda _path: None)
    monkeypatch.setattr(sandbox.os.path, "realpath", lambda path: path)
    original_isdir = sandbox.os.path.isdir
    monkeypatch.setattr(
        sandbox.os.path,
        "isdir",
        lambda path: path == developer or original_isdir(path),
    )
    profile = sandbox._macos_seatbelt_profile("/tmp/work")

    assert f'(subpath "{developer}")' in profile


def test_macos_profile_denies_writes_to_nested_runtime(tmp_path, monkeypatch):
    sandbox = _load_sandbox_module()
    workdir = tmp_path / "project"
    runtime = workdir / ".venv"
    runtime.mkdir(parents = True)
    wd = os.path.realpath(workdir)
    rp = os.path.realpath(runtime)

    monkeypatch.setattr(sandbox, "_python_read_paths", lambda: [rp])
    monkeypatch.setattr(sandbox, "_safe_subpath", lambda path: path.replace("\\", "/"))
    profile = sandbox._macos_seatbelt_profile(str(workdir))

    expected = f'(deny file-write* file-ioctl\n    (subpath "{rp.replace(chr(92), "/")}")\n)'
    assert f'(allow file-write* (subpath "{wd.replace(chr(92), "/")}"))' in profile
    assert expected in profile


def test_macos_ca_reads_exclude_private_key_directories(monkeypatch):
    sandbox = _load_sandbox_module()
    monkeypatch.setattr(sandbox, "_python_read_paths", lambda: [])
    monkeypatch.setattr(sandbox, "_assert_no_external_hardlinks", lambda _path: None)
    monkeypatch.setattr(sandbox.os.path, "realpath", lambda path: path)
    profile = sandbox._macos_seatbelt_profile("/tmp/work")

    assert '(subpath "/private/etc/ssl")' not in profile
    assert "/private/etc/ssl/private" not in profile
    assert '(literal "/private/etc/ssl/cert.pem")' in profile
    assert '(subpath "/private/etc/ssl/certs")' in profile

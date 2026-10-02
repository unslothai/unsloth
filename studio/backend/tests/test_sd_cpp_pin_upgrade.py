# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed sd.cpp install made for an older pin is upgraded to the pin this Studio ships.

Before this, ensure_sd_cpp_binary / ensure_sd_server_binary reinstalled only a missing, unrunnable or
wrong-accelerator binary, so moving DEFAULT_TAG reached new installs only: a host that installed the
2026-08-09 u13b9d92 bundle kept running it after the pin moved to u1d02858 (observed on a benchmark
host whose Studio pinned u1d02858 while ~/.unsloth/stable-diffusion.cpp still held u13b9d92).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_STUDIO = Path(__file__).resolve().parents[2]
if str(_STUDIO) not in sys.path:
    sys.path.insert(0, str(_STUDIO))

import install_sd_cpp_prebuilt as sdmod  # noqa: E402

_CLI = "sd-cli.exe" if sys.platform == "win32" else "sd-cli"
_SERVER = "sd-server.exe" if sys.platform == "win32" else "sd-server"


def _tree(tmp_path, monkeypatch, record):
    import core.inference.sd_cpp_backend as bk

    root = tmp_path / "sd-home" / "stable-diffusion.cpp"
    (root / "sd-bin").mkdir(parents = True)
    (root / ".unsloth-studio-owned").touch()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "sd-home" / "studio"))
    monkeypatch.delenv("UNSLOTH_SD_CPP_TAG", raising = False)
    monkeypatch.delenv("UNSLOTH_SD_CPP_AUTO_UPGRADE", raising = False)
    if record is not None:
        (root / sdmod.INSTALL_RECORD).write_text(json.dumps(record), encoding = "utf-8")
    sdmod._INSTALLED_ACCELERATOR_MEMO.clear()
    sdmod._INSTALLED_SHIPS_SERVER_MEMO.clear()
    cli = root / "sd-bin" / _CLI
    cli.write_bytes(b"old-build")
    server = root / "sd-bin" / _SERVER
    server.write_bytes(b"old-build")
    monkeypatch.setattr(bk, "find_sd_cpp_binary", lambda: str(cli))
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: str(server))
    monkeypatch.setattr(bk, "_usable_or_discard_managed", lambda *_a, **_k: True)
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(bk, "_failed_accelerator_upgrades", set())
    monkeypatch.setattr(bk, "_failed_pin_upgrades", set())
    return bk, root, cli, server


def _recording_install(root, cli, server, installs):
    def _install(**kwargs):
        installs.append(kwargs)
        cli.write_bytes(b"new-build")
        server.write_bytes(b"new-build")
        sdmod._write_install_record(
            root, accelerator = kwargs["accelerator"], repo = "r", tag = sdmod.DEFAULT_TAG
        )
        return cli

    return _install


OLD = "master-813-bfbef5b-u13b9d92"


def test_an_install_for_an_older_pin_is_upgraded_once(tmp_path, monkeypatch):
    bk, root, cli, server = _tree(
        tmp_path, monkeypatch, {"accelerator": "cuda", "repo": "r", "tag": OLD}
    )
    assert OLD != sdmod.DEFAULT_TAG
    installs: list = []
    monkeypatch.setattr(sdmod, "install", _recording_install(root, cli, server, installs))

    assert bk.ensure_sd_cpp_binary(accelerator = "cuda") == str(cli)
    assert [k["accelerator"] for k in installs] == ["cuda"]
    assert cli.read_bytes() == b"new-build"
    # The record now names the shipped pin: the next load reuses it.
    assert bk.ensure_sd_cpp_binary(accelerator = "cuda") == str(cli)
    assert bk.ensure_sd_server_binary(accelerator = "cuda") == str(server)
    assert len(installs) == 1


def test_an_unwritable_record_does_not_redownload_on_every_load(tmp_path, monkeypatch):
    bk, root, cli, server = _tree(
        tmp_path, monkeypatch, {"accelerator": "cuda", "repo": "r", "tag": OLD}
    )
    import builtins

    def _open(
        path,
        mode = "r",
        *args,
        **kwargs,
    ):
        if "w" in mode and Path(path).name == sdmod.INSTALL_RECORD:
            raise PermissionError("record held by another writer")
        return builtins.open(path, mode, *args, **kwargs)

    monkeypatch.setattr(sdmod, "open", _open, raising = False)
    installs: list = []
    monkeypatch.setattr(sdmod, "install", _recording_install(root, cli, server, installs))
    for _ in range(3):
        assert bk.ensure_sd_cpp_binary(accelerator = "cuda") == str(cli)
    assert bk.ensure_sd_server_binary(accelerator = "cuda") == str(server)
    assert len(installs) == 1
    assert OLD in (root / sdmod.INSTALL_RECORD).read_text(encoding = "utf-8")


def test_the_server_resolver_upgrades_an_old_pin_too(tmp_path, monkeypatch):
    bk, root, cli, server = _tree(
        tmp_path, monkeypatch, {"accelerator": "cuda", "repo": "r", "tag": OLD}
    )
    installs: list = []
    monkeypatch.setattr(sdmod, "install", _recording_install(root, cli, server, installs))
    assert bk.ensure_sd_server_binary(accelerator = "cuda") == str(server)
    assert server.read_bytes() == b"new-build"
    assert len(installs) == 1


def test_the_current_pin_is_never_reinstalled(tmp_path, monkeypatch):
    bk, root, cli, server = _tree(tmp_path, monkeypatch, None)
    sdmod._write_install_record(root, accelerator = "cuda", repo = "r", tag = sdmod.DEFAULT_TAG)

    def _install(**_kwargs):
        raise AssertionError("the shipped pin must not be reinstalled")

    monkeypatch.setattr(sdmod, "install", _install)
    assert bk.ensure_sd_cpp_binary(accelerator = "cuda") == str(cli)
    assert bk.ensure_sd_server_binary(accelerator = "cuda") == str(server)


@pytest.mark.parametrize(
    "record",
    [
        None,  # predates the record
        {"accelerator": "cuda", "repo": "r"},  # no tag
        {"accelerator": "cuda", "repo": "r", "tag": ""},
    ],
)
def test_an_install_that_cannot_say_its_pin_is_left_alone(tmp_path, monkeypatch, record):
    bk, root, cli, server = _tree(tmp_path, monkeypatch, record)

    def _install(**_kwargs):
        raise AssertionError("unknown is not stale")

    monkeypatch.setattr(sdmod, "install", _install)
    # cpu: an unrecorded install is not an accelerator mismatch either, so only the pin check could fire.
    assert bk.ensure_sd_cpp_binary(accelerator = "cpu") == str(cli)


def test_a_fallback_install_for_the_current_pin_is_not_redownloaded(tmp_path, monkeypatch):
    """A host the mirror does not build gets the upstream release (or latest), whose tag can never equal the pin.
    requested_tag records what the install was FOR, so it is not re-fetched on every load."""
    bk, root, cli, server = _tree(
        tmp_path,
        monkeypatch,
        {
            "accelerator": "vulkan",
            "repo": "leejet/stable-diffusion.cpp",
            "tag": "master-999-abcdef0",
            "requested_tag": sdmod.DEFAULT_TAG,
        },
    )

    def _install(**_kwargs):
        raise AssertionError("a fallback install for the current pin is current")

    monkeypatch.setattr(sdmod, "install", _install)
    assert bk.ensure_sd_cpp_binary(accelerator = "vulkan") == str(cli)


def test_an_old_record_holding_the_upstream_form_of_the_pin_is_current(tmp_path, monkeypatch):
    bk, root, cli, server = _tree(
        tmp_path,
        monkeypatch,
        {"accelerator": "vulkan", "repo": "r", "tag": sdmod.upstream_tag_for(sdmod.DEFAULT_TAG)},
    )
    monkeypatch.setattr(
        sdmod, "install", lambda **_k: (_ for _ in ()).throw(AssertionError("current"))
    )
    assert bk.ensure_sd_cpp_binary(accelerator = "vulkan") == str(cli)


def test_kill_switch_keeps_the_installed_bundle(tmp_path, monkeypatch):
    bk, root, cli, server = _tree(
        tmp_path, monkeypatch, {"accelerator": "cuda", "repo": "r", "tag": OLD}
    )
    monkeypatch.setenv("UNSLOTH_SD_CPP_AUTO_UPGRADE", "0")
    monkeypatch.setattr(
        sdmod, "install", lambda **_k: (_ for _ in ()).throw(AssertionError("kill switch"))
    )
    assert bk.ensure_sd_cpp_binary(accelerator = "cuda") == str(cli)
    assert bk.ensure_sd_server_binary(accelerator = "cuda") == str(server)


def test_tracking_latest_never_counts_as_a_moved_pin(tmp_path, monkeypatch):
    bk, root, cli, server = _tree(
        tmp_path, monkeypatch, {"accelerator": "cuda", "repo": "r", "tag": OLD}
    )
    monkeypatch.setenv("UNSLOTH_SD_CPP_TAG", "")
    monkeypatch.setattr(
        sdmod, "install", lambda **_k: (_ for _ in ()).throw(AssertionError("latest"))
    )
    assert bk.ensure_sd_cpp_binary(accelerator = "cuda") == str(cli)


def test_a_failed_pin_upgrade_keeps_the_old_binary_and_stops_retrying(tmp_path, monkeypatch):
    bk, root, cli, server = _tree(
        tmp_path, monkeypatch, {"accelerator": "cuda", "repo": "r", "tag": OLD}
    )
    attempts: list = []

    def _install(**kwargs):
        attempts.append(kwargs)
        raise RuntimeError("offline")

    monkeypatch.setattr(sdmod, "install", _install)
    assert bk.ensure_sd_cpp_binary(accelerator = "cuda") == str(cli)
    assert bk.ensure_sd_cpp_binary(accelerator = "cuda") == str(cli)
    assert bk.ensure_sd_server_binary(accelerator = "cuda") == str(server)
    assert len(attempts) == 1, "a hopeless upgrade is attempted once per process, not once per load"
    assert cli.read_bytes() == b"old-build"


def test_a_failed_upgrade_for_one_accelerator_still_lets_another_upgrade_the_pin(
    tmp_path, monkeypatch
):
    bk, root, cli, server = _tree(
        tmp_path, monkeypatch, {"accelerator": "cpu", "repo": "r", "tag": OLD}
    )
    installs: list = []
    ok = _recording_install(root, cli, server, installs)

    def _install(**kwargs):
        if kwargs["accelerator"] == "cuda":
            installs.append(kwargs)
            raise RuntimeError("no CUDA asset for this host")
        return ok(**kwargs)

    monkeypatch.setattr(sdmod, "install", _install)
    assert bk.ensure_sd_cpp_binary(accelerator = "cuda") == str(cli)
    assert cli.read_bytes() == b"old-build"
    assert bk.ensure_sd_cpp_binary(accelerator = "cpu") == str(cli)
    assert [k["accelerator"] for k in installs] == ["cuda", "cpu"]
    assert cli.read_bytes() == b"new-build"


def test_no_upgrade_while_the_managed_tree_is_in_use(tmp_path, monkeypatch):
    bk, root, cli, server = _tree(
        tmp_path, monkeypatch, {"accelerator": "cuda", "repo": "r", "tag": OLD}
    )
    monkeypatch.setattr(bk, "_managed_tree_in_use", lambda: True)
    monkeypatch.setattr(
        sdmod, "install", lambda **_k: (_ for _ in ()).throw(AssertionError("in use"))
    )
    assert bk.ensure_sd_cpp_binary(accelerator = "cuda") == str(cli)


def test_a_user_supplied_binary_is_never_upgraded(tmp_path, monkeypatch):
    bk, root, cli, server = _tree(
        tmp_path, monkeypatch, {"accelerator": "cuda", "repo": "r", "tag": OLD}
    )
    (root / ".unsloth-studio-owned").unlink()
    monkeypatch.setattr(
        sdmod, "install", lambda **_k: (_ for _ in ()).throw(AssertionError("not ours"))
    )
    assert bk.ensure_sd_cpp_binary(accelerator = "cuda") == str(cli)


def test_new_records_carry_the_pin_they_were_made_for(tmp_path, monkeypatch):
    monkeypatch.delenv("UNSLOTH_SD_CPP_TAG", raising = False)
    sdmod._write_install_record(tmp_path, accelerator = "cuda", repo = "r", tag = "whatever-it-got")
    rec = sdmod.read_install_record(tmp_path)
    assert rec["requested_tag"] == sdmod.DEFAULT_TAG
    assert sdmod.install_is_stale(tmp_path) is False
    monkeypatch.setenv("UNSLOTH_SD_CPP_TAG", "master-999-0000000-u0000000")
    assert sdmod.install_is_stale(tmp_path) is True

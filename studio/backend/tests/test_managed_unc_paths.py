# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's path checks refuse a UNC or device path before touching the disk."""

import errno
import os
from pathlib import Path

import pytest
from fastapi import HTTPException

from core.training import account_jobs
from hub.services.models import account_access
from utils.account_context import AccountContext, bind_account, reset_account

ALICE = AccountContext("a" * 32, "alice")
UNC = "\\\\attacker.invalid\\share\\m"
# Win32 hands a \\??\\ path to the NT namespace unchanged, so this is a UNC share too.
NT_UNC = "\\??\\UNC\\attacker.invalid\\share\\m"
UNC_PREFIXES = ("\\\\", "//", "\\??\\")


@pytest.fixture
def studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    return tmp_path


@pytest.fixture
def unc_calls(monkeypatch):
    calls = []
    real_exists, real_realpath, real_resolve = os.path.exists, os.path.realpath, Path.resolve
    real_stat = Path.stat

    def exists(p):
        if str(p).startswith(UNC_PREFIXES):
            calls.append(("exists", str(p)))
            return False
        return real_exists(p)

    def realpath(p, *a, **k):
        if str(p).startswith(UNC_PREFIXES):
            calls.append(("realpath", str(p)))
            return str(p)
        return real_realpath(p, *a, **k)

    def resolve(self, *a, **k):
        if str(self).startswith(UNC_PREFIXES):
            calls.append(("resolve", str(self)))
            return self
        return real_resolve(self, *a, **k)

    def stat(self, *a, **k):
        if str(self).startswith(UNC_PREFIXES):
            calls.append(("stat", str(self)))
            raise FileNotFoundError(errno.ENOENT, "not found", str(self))
        return real_stat(self, *a, **k)

    monkeypatch.setattr(os.path, "exists", exists)
    monkeypatch.setattr(os.path, "realpath", realpath)
    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(Path, "stat", stat)
    return calls


@pytest.mark.parametrize("unc", [UNC, NT_UNC])
def test_account_checks_refuse_a_unc_path_before_resolving_it(studio_home, unc_calls, unc):
    token = bind_account(ALICE)
    try:
        for reference in (False, True):
            with pytest.raises(HTTPException) as info:
                account_jobs.account_path(unc, reference = reference)
            assert info.value.status_code == 403
        assert account_access.model_visible(unc) is False
    finally:
        reset_account(token)
    assert unc_calls == []


def test_a_path_inside_the_workspace_still_passes(studio_home):
    from utils.paths.storage_roots import workspace_root
    token = bind_account(ALICE)
    try:
        inside = workspace_root() / "data" / "file.jsonl"
        assert account_jobs.account_path(str(inside)) == str(inside)
    finally:
        reset_account(token)


@pytest.mark.parametrize(
    "value",
    [
        "\\\\host\\share",
        "//host/share",
        "\\\\?\\C:\\x",
        "\\\\.\\PhysicalDrive0",
        "\\/host/share",
        "\\??\\UNC\\host\\share",
    ],
)
def test_network_and_device_spellings_are_recognised(value):
    from utils.paths.path_utils import is_network_or_device_path
    assert is_network_or_device_path(value)


def test_local_spellings_are_not_network_paths():
    from utils.paths.path_utils import is_network_or_device_path
    assert not is_network_or_device_path("C:\\Users\\x")
    assert not is_network_or_device_path("/home/x")


def test_the_allowlist_fallback_does_not_touch_a_unc_path(studio_home, unc_calls, monkeypatch):
    # The allowlist check resolves paths on disk, which a share path must never reach.
    monkeypatch.setattr(account_access, "model_allowlist", lambda: {"org/allowed"})
    token = bind_account(ALICE)
    try:
        with pytest.raises(HTTPException) as info:
            account_access.require_model_access(UNC)
        assert info.value.status_code == 404
    finally:
        reset_account(token)
    assert unc_calls == []


def test_a_folder_scan_refuses_a_unc_path_before_resolving_it(studio_home, unc_calls):
    token = bind_account(ALICE)
    try:
        with pytest.raises(HTTPException) as info:
            account_access.private_directory(UNC, "models")
        assert info.value.status_code == 404
    finally:
        reset_account(token)
    assert unc_calls == []


@pytest.mark.parametrize(
    "field", ["model_path", "transformer_prequant_path", "text_encoder_file", "vae_file"]
)
def test_media_references_refuse_a_unc_path_before_resolving_it(studio_home, unc_calls, field):
    from types import SimpleNamespace

    request = SimpleNamespace(model_path = "org/model", adapters = None)
    setattr(request, field, [UNC] if field == "text_encoder_file" else UNC)
    token = bind_account(ALICE)
    try:
        with pytest.raises(HTTPException) as info:
            account_access.require_media_references(request)
        assert info.value.status_code == 404
    finally:
        reset_account(token)
    assert unc_calls == []


def test_a_diffusion_data_dir_refuses_a_unc_path_before_resolving_it(studio_home, unc_calls):
    from routes import training

    token = bind_account(ALICE)
    try:
        with pytest.raises(HTTPException) as info:
            training._resolve_diffusion_data_dir(UNC)
        assert info.value.status_code == 403
    finally:
        reset_account(token)
    assert unc_calls == []

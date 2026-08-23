# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import stat
from types import SimpleNamespace

import pytest

from hub.utils import paths


_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


def _windows_stat(mode: int, reparse_tag: int) -> SimpleNamespace:
    return SimpleNamespace(
        st_mode = mode,
        st_file_attributes = _FILE_ATTRIBUTE_REPARSE_POINT,
        st_reparse_tag = reparse_tag,
    )


@pytest.mark.parametrize("reparse_tag", [0x9000001A, 0x80000013])
def test_data_transparent_reparse_points_are_not_redirects(monkeypatch, reparse_tag):
    monkeypatch.setattr(
        paths.stat,
        "FILE_ATTRIBUTE_REPARSE_POINT",
        _FILE_ATTRIBUTE_REPARSE_POINT,
        raising = False,
    )

    assert paths.is_redirect_stat(_windows_stat(stat.S_IFREG, reparse_tag)) is False


@pytest.mark.parametrize("reparse_tag", [0xA000000C, 0xA0000003])
def test_name_surrogate_reparse_points_are_redirects(reparse_tag):
    assert paths.is_redirect_stat(_windows_stat(stat.S_IFDIR, reparse_tag)) is True


def test_app_execution_links_are_redirects(monkeypatch):
    app_exec_link_tag = 0x8000001B
    monkeypatch.setattr(
        paths.stat,
        "IO_REPARSE_TAG_APPEXECLINK",
        app_exec_link_tag,
        raising = False,
    )

    assert paths.is_redirect_stat(_windows_stat(stat.S_IFREG, app_exec_link_tag)) is True


def test_posix_symlinks_are_redirects_without_a_reparse_tag():
    assert paths.is_redirect_stat(SimpleNamespace(st_mode = stat.S_IFLNK)) is True

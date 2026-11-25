# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Worst-case inputs for the python / terminal safety screen: same decisions, bounded work."""

from __future__ import annotations

import pytest

from core.inference import tools

_SESSION = "safety-screen-worst-case"


@pytest.fixture
def studio_home(monkeypatch, tmp_path):
    home = tmp_path / "studio-home"
    (home / "auth").mkdir(parents = True)
    (home / "sandbox" / _SESSION).mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    monkeypatch.setattr(tools, "_studio_auth_markers_cache", None)
    try:
        yield home
    finally:
        tools._studio_auth_markers_cache = None


def _count_credential_scans(monkeypatch):
    calls = [0]
    real = tools._references_studio_credential_here

    def counted(*args, **kwargs):
        calls[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(tools, "_references_studio_credential_here", counted)
    return counted, calls


@pytest.mark.parametrize("depth", [10, 64, 65, 130, 200, 300])
def test_nested_brackets_scan_each_rewrite_once(studio_home, monkeypatch, depth):
    # Both bracket rewrites reach the same texts by many paths: depth 200 took 143,571 scans.
    scan, calls = _count_credential_scans(monkeypatch)
    assert scan("x = " + "[" * depth + "]" * depth, None) is False
    assert calls[0] <= 2 * depth * depth // 50 + 4 * depth, calls[0]


def test_nested_brackets_with_a_workdir_and_a_credential(studio_home, monkeypatch):
    workdir = str(studio_home / "sandbox" / _SESSION)
    scan, calls = _count_credential_scans(monkeypatch)
    nest = "[" * 150 + "]" * 150
    assert scan(f"echo {nest}; cat ../../aut[h]/aut[h].db", workdir) is True
    assert scan(f"echo {nest}; cat {studio_home}/aut[h]/[a]uth.db", None) is True
    assert scan(f"echo {nest}; cat ./notes[1].txt", workdir) is False
    assert calls[0] < 20000, calls[0]


def test_the_stack_still_runs_out_where_it_did(studio_home):
    # Kept per depth and per call: the bound on nesting (and its RecursionError) is unchanged.
    with pytest.raises(RecursionError):
        tools._references_studio_credential_here("x = " + "[" * 2000 + "]" * 2000, None)


def test_a_clean_scan_is_not_kept_across_calls(studio_home, monkeypatch):
    workdir = str(studio_home / "sandbox" / _SESSION)
    text = "cat ../../aut[h]/aut[h].db"
    assert tools._references_studio_credential_here(text, None) is False
    assert tools._references_studio_credential_here(text, workdir) is True
    assert tools._references_studio_credential_here(text, None) is False

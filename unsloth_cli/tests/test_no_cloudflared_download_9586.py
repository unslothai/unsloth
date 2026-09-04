# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for issue #9586 channel 3a: a unit test downloading cloudflared.

The `--secure` path asks whether a tunnel could start, and that question is answered by
`ensure_cloudflared()`, which FETCHES a ~40 MB binary when none is found. The `unsloth run`
re-exec tests in this root reach it for real, so the suite depended on network reachability
and wrote a large file outside anything it owns.

The guard is a stub on PATH, not a patched module, because the caller execs a fresh copy of
`cloudflare_tunnel` by file path on every call -- see the conftest fixture for the
measurement behind that.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


_BACKEND = Path(__file__).resolve().parents[2] / "studio" / "backend"


def _cloudflare_tunnel():
    if str(_BACKEND) not in sys.path:
        sys.path.insert(0, str(_BACKEND))
    import cloudflare_tunnel
    return cloudflare_tunnel


def test_a_cloudflared_is_visible_on_path():
    """The guard itself. Without it `which` finds nothing and the fetch path opens."""
    found = shutil.which("cloudflared")
    assert found, "conftest should have put a cloudflared stub on PATH"
    assert "fake-bin" in found, found


def test_find_cloudflared_returns_the_stub_rather_than_reaching_the_cache():
    """`find_cloudflared` consults PATH first, which is what makes the guard work.

    Both halves asserted non-empty. Comparing the two results alone passes vacuously
    without the guard -- `which` finds nothing, `find_cloudflared` has nothing cached
    yet, and `None == None` holds while the very next test performs the download.
    """
    ct = _cloudflare_tunnel()

    on_path = shutil.which("cloudflared")
    assert on_path, "conftest should have put a cloudflared stub on PATH"

    found = ct.find_cloudflared()
    assert found, "find_cloudflared should resolve the PATH stub"
    assert found == on_path


def test_ensure_cloudflared_returns_without_fetching():
    """The real function, unpatched, taking its early-return branch.

    Nothing here is stubbed: the lookup, the cache path and the platform asset logic all
    run. Only the *outcome* differs, because a cloudflared is already visible.
    """
    ct = _cloudflare_tunnel()

    assert ct.ensure_cloudflared() == shutil.which("cloudflared")


def test_no_cloudflared_binary_is_written_into_the_studio_bin_root():
    """The observable the issue reports: ~39.8 MB appearing under the studio home."""
    ct = _cloudflare_tunnel()

    cached = ct._cache_path()
    if cached is None:
        return
    assert not cached.exists(), f"a test fetched cloudflared to {cached}"


def test_the_guard_survives_a_freshly_exec_d_copy_of_the_module():
    """The property a patched module object would NOT have.

    `unsloth_cli/commands/studio.py` loads cloudflare_tunnel with
    `spec_from_file_location(...)` and execs it anew on every call, so a `monkeypatch`
    against an imported copy reaches a different object and the download still happens.
    A PATH stub satisfies any copy.
    """
    import importlib.util

    tunnel_py = _BACKEND / "cloudflare_tunnel.py"
    if not tunnel_py.is_file():
        return

    if str(_BACKEND) not in sys.path:
        sys.path.insert(0, str(_BACKEND))
    spec = importlib.util.spec_from_file_location("studio.backend.cloudflare_tunnel", tunnel_py)
    assert spec is not None and spec.loader is not None
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)

    assert fresh.ensure_cloudflared() == shutil.which("cloudflared")
    cached = fresh._cache_path()
    if cached is not None:
        assert not cached.exists(), f"the fresh copy fetched cloudflared to {cached}"

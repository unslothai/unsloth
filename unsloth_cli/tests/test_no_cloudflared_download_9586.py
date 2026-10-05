# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#9586: the --secure cloudflared check must not download the binary in tests."""

import importlib.util
import shutil
import urllib.request
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[2] / "studio" / "backend"


def test_fresh_tunnel_module_resolves_the_path_stub_without_downloading(monkeypatch):
    def no_download(*args, **kwargs):
        raise AssertionError("cloudflared download attempted")

    monkeypatch.setattr(urllib.request, "urlopen", no_download)
    monkeypatch.syspath_prepend(str(_BACKEND))
    # Load it the way _tunnel_binary_confirmed_unavailable does, as a fresh module.
    spec = importlib.util.spec_from_file_location(
        "studio.backend.cloudflare_tunnel", _BACKEND / "cloudflare_tunnel.py"
    )
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)

    stub = shutil.which("cloudflared")
    assert stub and "fake-bin" in stub
    # Must be the stub, not a binary a developer already has cached in their studio home.
    assert fresh.ensure_cloudflared() == stub

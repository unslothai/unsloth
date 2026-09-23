# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("playwright_mtp_download_visibility.py")


def _fixture_module():
    spec = importlib.util.spec_from_file_location("playwright_mtp_download_visibility", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_large_cache_fixture_reports_size_without_allocating_it(tmp_path):
    module = _fixture_module()
    target = tmp_path / "main.gguf"
    target.write_bytes(b"m")
    logical = module._LogicalGgufPath(target, module.MAIN_BYTES)

    assert target.stat().st_size == 1
    assert logical.stat().st_size == module.MAIN_BYTES
    assert logical.relative_to(tmp_path) == Path("main.gguf")


def test_vite_launch_resolves_the_platform_npm_executable():
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'npm = shutil.which("npm") or "npm"' in source
    assert '[\n            "npm",' not in source

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import re
from fnmatch import fnmatchcase
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_generated_compiled_caches_are_excluded():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding = "utf-8")
    finder = text.split("[tool.setuptools.packages.find]", 1)[1].split("\n[", 1)[0]
    exclude = re.search(
        r"^exclude\s*=\s*\[(.*?)\]",
        finder,
        re.MULTILINE | re.DOTALL,
    )
    assert exclude, "no packages.find exclude list in pyproject.toml"
    patterns = re.findall(r'["\']([^"\']+)["\']', exclude.group(1))

    for package in ("unsloth_compiled_cache", "studio.backend.unsloth_compiled_cache"):
        assert any(fnmatchcase(package, pattern) for pattern in patterns)

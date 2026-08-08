# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from pathlib import Path

from setuptools.config.pyprojecttoml import read_configuration
from setuptools.discovery import PEP420PackageFinder


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_nested_compiled_cache_is_excluded(tmp_path):
    config = read_configuration(str(REPO_ROOT / "pyproject.toml"), expand = False)
    finder = config["tool"]["setuptools"]["packages"]["find"]
    cache = tmp_path / "studio" / "backend" / "unsloth_compiled_cache"
    cache.mkdir(parents = True)
    (cache / "moe_utils.py").write_text("VALUE = 1\n", encoding = "utf-8")

    packages = PEP420PackageFinder.find(
        str(tmp_path),
        include = finder["include"],
        exclude = finder["exclude"],
    )

    assert "studio.backend.unsloth_compiled_cache" not in packages

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The studio extra must stay a mirror of studio/backend/requirements/studio.txt.

`pip install "unsloth[studio]"` is the only way to get the server stack without
install.sh, so it has to describe the same environment. Nothing else keeps them
in sync: the extra is hand-maintained while the installer reads the requirements
file, and drift reintroduces #4701 / #5260 / #7147.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PYPROJECT = REPO_ROOT / "pyproject.toml"
STUDIO_TXT = REPO_ROOT / "studio" / "backend" / "requirements" / "studio.txt"

# Reached by `unsloth train` / `export` / `chat` / `inference` / `studio` at
# import time even with no server involved, so a plain `pip install unsloth`
# must provide them. Keep in step with unsloth_cli/_studio_deps.py.
CORE_RUNTIME_PACKAGES = ("structlog",)


def _load_pyproject() -> dict:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        tomllib = pytest.importorskip("tomli")
    return tomllib.loads(PYPROJECT.read_text(encoding = "utf-8"))


def _requirement_lines(path: pathlib.Path) -> list[str]:
    out = []
    for line in path.read_text(encoding = "utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if text and not text.startswith("-"):
            out.append(text)
    return out


def _normalise(name: str) -> str:
    """PEP 503 name normalisation, so pyjwt/PyJWT and nest_asyncio/nest-asyncio match."""
    head = name
    for sep in ("===", "==", ">=", "<=", "~=", "!=", ">", "<", "[", ";", " "):
        idx = head.find(sep)
        if idx > 0:
            head = head[:idx]
    return head.strip().lower().replace("_", "-").replace(".", "-")


def test_studio_extra_exists():
    extras = _load_pyproject()["project"]["optional-dependencies"]
    assert "studio" in extras, (
        "pyproject.toml has no `studio` extra. The wheel ships studio/ and "
        "studio.backend*, so there must be a supported way to pip-install the "
        "dependencies those modules import."
    )


def test_studio_extra_matches_requirements_file():
    extras = _load_pyproject()["project"]["optional-dependencies"]
    extra = sorted(_normalise(entry) for entry in extras["studio"])
    required = sorted(_normalise(entry) for entry in _requirement_lines(STUDIO_TXT))

    missing = sorted(set(required) - set(extra))
    surplus = sorted(set(extra) - set(required))
    assert not missing, (
        f"studio.txt lists {missing} but the `studio` extra does not. "
        "`pip install \"unsloth[studio]\"` would build a venv the Studio server "
        "cannot boot in. Add them to [project.optional-dependencies] studio."
    )
    assert not surplus, (
        f"The `studio` extra lists {surplus} but studio.txt does not. "
        "Remove them, or add them to studio.txt if install.sh needs them too."
    )


@pytest.mark.parametrize("package", CORE_RUNTIME_PACKAGES)
def test_cli_runtime_packages_are_core_dependencies(package):
    core = [_normalise(entry) for entry in _load_pyproject()["project"]["dependencies"]]
    assert _normalise(package) in core, (
        f"{package} is imported at module scope by the studio.backend chain that "
        f"`unsloth train` / `unsloth export` walk, so a plain `pip install unsloth` "
        f"must provide it. Without it those commands fail with "
        f"ModuleNotFoundError before doing any work."
    )

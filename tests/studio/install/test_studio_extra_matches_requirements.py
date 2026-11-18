# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Nothing else keeps the studio extra in sync with studio/backend/requirements/studio.txt."""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PYPROJECT = REPO_ROOT / "pyproject.toml"
STUDIO_TXT = REPO_ROOT / "studio" / "backend" / "requirements" / "studio.txt"

# Imported at module scope by every CLI command: structlog via studio.backend, click via start.py.
CORE_RUNTIME_PACKAGES = ("structlog", "click")


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
    """PEP 503 normalisation, so PyJWT/pyjwt and nest_asyncio/nest-asyncio match."""
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
        "studio.backend*, so their dependencies need a pip-installable home."
    )


def test_s3_sdk_is_only_in_the_s3_extra():
    from packaging.requirements import Requirement

    project = _load_pyproject()["project"]
    extras = project["optional-dependencies"]
    assert extras["s3"] == ["boto3>=1.34.0"]
    aws_packages = {"boto3", "botocore", "s3transfer", "jmespath"}
    default_requirements = (
        project["dependencies"] + extras["studio"] + _requirement_lines(STUDIO_TXT)
    )
    assert not aws_packages.intersection(
        Requirement(entry).name.lower() for entry in default_requirements
    )


def test_huggingface_extra_contains_no_diffusers_vcs_reference():
    from packaging.requirements import Requirement

    entries = _load_pyproject()["project"]["optional-dependencies"]["huggingfacenotorch"]
    diffusers = [Requirement(entry) for entry in entries if Requirement(entry).name == "diffusers"]
    assert diffusers
    assert all(requirement.url is None for requirement in diffusers), (
        "PyPI rejects VCS direct references in uploaded Requires-Dist metadata; "
        "keep temporary Diffusers commit pins in Unsloth installer requirement files."
    )


def test_studio_extra_matches_requirements_file():
    extras = _load_pyproject()["project"]["optional-dependencies"]
    extra = sorted(_normalise(entry) for entry in extras["studio"])
    required = sorted(_normalise(entry) for entry in _requirement_lines(STUDIO_TXT))

    missing = sorted(set(required) - set(extra))
    surplus = sorted(set(extra) - set(required))
    assert not missing, (
        f"studio.txt lists {missing} but the `studio` extra does not. "
        '`pip install "unsloth[studio]"` would build a venv the Unsloth server '
        "cannot boot in. Add them to [project.optional-dependencies] studio."
    )
    assert not surplus, (
        f"The `studio` extra lists {surplus} but studio.txt does not. "
        "Remove them, or add them to studio.txt if install.sh needs them too."
    )


def _specs(entries: list[str]) -> set[tuple[str, str, str]]:
    """(name, specifier, marker) per entry, quote style and ordering normalised."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    out = set()
    for entry in entries:
        requirement = Requirement(entry)
        specifier = ",".join(sorted(str(s) for s in requirement.specifier))
        out.add(
            (
                canonicalize_name(requirement.name),
                specifier,
                str(requirement.marker) if requirement.marker else "",
            )
        )
    return out


def test_studio_extra_matches_requirement_versions():
    """Compare versions and markers, not just names: bare extras let pip resolve a different stack."""
    extras = _load_pyproject()["project"]["optional-dependencies"]
    extra = _specs(extras["studio"])
    required = _specs(_requirement_lines(STUDIO_TXT))

    missing = sorted(required - extra)
    surplus = sorted(extra - required)
    assert not missing, (
        f"studio.txt specifies {missing} but the `studio` extra does not. "
        '`pip install "unsloth[studio]"` would resolve a different stack than '
        "install.sh builds. Mirror the specifier and marker, not just the name."
    )
    assert not surplus, (
        f"The `studio` extra specifies {surplus} but studio.txt does not. "
        "Mirror studio.txt exactly, or update studio.txt if the installer needs it."
    )


@pytest.mark.parametrize("package", CORE_RUNTIME_PACKAGES)
def test_cli_runtime_packages_are_core_dependencies(package):
    core = [_normalise(entry) for entry in _load_pyproject()["project"]["dependencies"]]
    assert _normalise(package) in core, (
        f"{package} is imported at module scope by the studio.backend chain "
        f"`unsloth train` / `unsloth export` walk, so a plain `pip install "
        f"unsloth` must provide it or they die with ModuleNotFoundError."
    )


def test_studio_declares_psutil_for_the_cluster_link():
    # The cluster link records, reaps and reads the connections of its ggml-rpc-server through psutil.
    text = (REPO_ROOT / "studio" / "backend" / "requirements" / "studio.txt").read_text(
        encoding = "utf-8"
    )
    assert any(line.split("==")[0].strip() == "psutil" for line in text.splitlines())

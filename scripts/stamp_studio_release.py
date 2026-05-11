#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stamp and verify display-only Studio release metadata for builds."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_INFO_PATH = (
    REPO_ROOT / "studio" / "backend" / "utils" / "_studio_release_build.py"
)
BUILD_INFO_SUFFIX = "studio/backend/utils/_studio_release_build.py"
VERSION_RE = re.compile(r"^v\d+\.\d+\.\d+(?:-[0-9A-Za-z.][0-9A-Za-z.-]*)?$")
GIT_DESCRIBE_SUFFIX_RE = re.compile(r"-\d+-g[0-9A-Fa-f]+(?:-dirty)?$")
MAX_VERSION_LENGTH = 64
PLACEHOLDER = """# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

\"\"\"Build-stamped Studio release metadata.

Release builds may rewrite this module in the build workspace before creating
Python artifacts. Keep the committed value neutral so source checkouts do not
accidentally report a stale release tag.
\"\"\"

STUDIO_RELEASE_VERSION = None
"""


def is_valid_version(value: object) -> bool:
    if not isinstance(value, str):
        return False
    version = value.strip()
    if not version or len(version) > MAX_VERSION_LENGTH:
        return False
    if version.endswith("-dirty") or GIT_DESCRIBE_SUFFIX_RE.search(version):
        return False
    return VERSION_RE.fullmatch(version) is not None


def _exact_git_tag() -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "describe",
                "--tags",
                "--exact-match",
                "--match",
                "v[0-9]*",
                "HEAD",
            ],
            cwd = REPO_ROOT,
            check = False,
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            timeout = 2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    tag = result.stdout.strip()
    return tag if is_valid_version(tag) else None


def _git_worktree_is_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd = REPO_ROOT,
            check = False,
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            timeout = 2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return True
    if result.returncode != 0:
        return True
    return bool(result.stdout.strip())


def _github_tag() -> str | None:
    if os.environ.get("GITHUB_REF_TYPE") != "tag":
        return None
    github_ref = os.environ.get("GITHUB_REF_NAME", "").strip()
    return github_ref or None


def resolve_version() -> tuple[str | None, str]:
    env_version = os.environ.get("UNSLOTH_STUDIO_RELEASE_VERSION", "").strip()
    if env_version:
        return (env_version, "UNSLOTH_STUDIO_RELEASE_VERSION")

    github_ref = _github_tag()
    if github_ref:
        return (github_ref, "GITHUB_REF_NAME")

    git_tag = _exact_git_tag()
    if git_tag:
        return (git_tag, "exact git tag")

    return (None, "none")


def build_info_source(version: str | None) -> str:
    literal = repr(version) if version is not None else "None"
    return f'''# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Build-stamped Studio release metadata."""

STUDIO_RELEASE_VERSION = {literal}
'''


def _env_version_conflicts(version: str) -> list[tuple[str, str]]:
    conflicts: list[tuple[str, str]] = []
    github_ref = _github_tag()
    if github_ref and is_valid_version(github_ref) and github_ref != version:
        conflicts.append(("GITHUB_REF_NAME", github_ref))

    git_tag = _exact_git_tag()
    if git_tag and git_tag != version:
        conflicts.append(("exact git tag", git_tag))

    return conflicts


def stamp(require_release: bool) -> int:
    version, source = resolve_version()
    if version is not None and not is_valid_version(version):
        print(
            f"Invalid Studio release version from {source}: {version!r}",
            file = sys.stderr,
        )
        return 2

    if version is not None and source == "UNSLOTH_STUDIO_RELEASE_VERSION":
        conflicts = _env_version_conflicts(version)
        if conflicts:
            details = ", ".join(f"{name}={value!r}" for name, value in conflicts)
            print(
                "UNSLOTH_STUDIO_RELEASE_VERSION does not match available "
                f"release tag metadata: {details}",
                file = sys.stderr,
            )
            return 2

    if require_release and source == "exact git tag" and _git_worktree_is_dirty():
        print(
            "Refusing to publish from a dirty exact-tag checkout. Set "
            "UNSLOTH_STUDIO_RELEASE_VERSION explicitly from release automation "
            "or publish from a clean tag checkout.",
            file = sys.stderr,
        )
        return 2

    if version is None:
        if require_release:
            print(
                "No Studio release version available. Set "
                "UNSLOTH_STUDIO_RELEASE_VERSION, build from a GitHub tag, "
                "or run from an exact local Studio release tag.",
                file = sys.stderr,
            )
            return 2
        BUILD_INFO_PATH.write_text(PLACEHOLDER, encoding = "utf-8")
        print("dev")
        return 0

    BUILD_INFO_PATH.write_text(build_info_source(version), encoding = "utf-8")
    print(f"Stamping Studio release version {version} from {source}", file = sys.stderr)
    print(version)
    return 0


def _read_wheel_member(path: Path) -> str | None:
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.endswith(BUILD_INFO_SUFFIX):
                return archive.read(name).decode("utf-8")
    return None


def _read_sdist_member(path: Path) -> str | None:
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            if member.name.endswith(BUILD_INFO_SUFFIX):
                extracted = archive.extractfile(member)
                if extracted is None:
                    return None
                return extracted.read().decode("utf-8")
    return None


def verify_dist(expected: str, dist_dir: Path) -> int:
    if not is_valid_version(expected):
        print(f"Invalid expected Studio release version: {expected!r}", file = sys.stderr)
        return 2

    artifacts = list(dist_dir.glob("*.whl")) + list(dist_dir.glob("*.tar.gz"))
    if not artifacts:
        print(f"No wheel or sdist artifacts found in {dist_dir}", file = sys.stderr)
        return 2

    expected_line = f"STUDIO_RELEASE_VERSION = {expected!r}"
    failures: list[str] = []
    for artifact in artifacts:
        if artifact.suffix == ".whl":
            content = _read_wheel_member(artifact)
        else:
            content = _read_sdist_member(artifact)
        if content is None:
            failures.append(f"{artifact.name}: missing {BUILD_INFO_SUFFIX}")
        elif expected_line not in content:
            failures.append(f"{artifact.name}: Studio release version mismatch")

    if failures:
        for failure in failures:
            print(failure, file = sys.stderr)
        return 2

    print(f"Verified Studio release version {expected} in {len(artifacts)} artifact(s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--require-release", action = "store_true")
    parser.add_argument("--verify-dist", type = Path)
    parser.add_argument("--expected")
    args = parser.parse_args()

    if args.verify_dist is not None:
        if not args.expected:
            parser.error("--verify-dist requires --expected")
        return verify_dist(args.expected, args.verify_dist)

    return stamp(require_release = args.require_release)


if __name__ == "__main__":
    raise SystemExit(main())

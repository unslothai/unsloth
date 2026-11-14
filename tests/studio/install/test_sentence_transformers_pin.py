# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import pathlib

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
REQ_ROOT = REPO_ROOT / "studio" / "backend" / "requirements"

TRANSFORMERS_WITHOUT_NESTED_GATHER = Version("5.2.0")
SENTENCE_TRANSFORMERS_WITH_FALLBACK = Version("5.2.3")


def _pins(path: pathlib.Path, name: str) -> dict[str, Version]:
    out = {}
    for line in path.read_text(encoding = "utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if not text or text.startswith("-"):
            continue
        try:
            req = Requirement(text)
        except InvalidRequirement:
            continue
        if canonicalize_name(req.name) != canonicalize_name(name):
            continue
        for spec in req.specifier:
            if spec.operator == "==":
                out[str(req.marker)] = Version(spec.version)
    return out


def _trains_with(transformers: Version, sentence_transformers: Version) -> bool:
    return (
        transformers < TRANSFORMERS_WITHOUT_NESTED_GATHER
        or sentence_transformers >= SENTENCE_TRANSFORMERS_WITH_FALLBACK
    )


def _requirement_files(root: pathlib.Path) -> list[pathlib.Path]:
    """The shipped requirement files. install_python_stack._filter_requirements writes a
    hidden `.<stem>-filtered-*.txt` copy beside its source and deletes it again, and tests
    that drive the installer do that in this tree on other xdist workers, so a dot-named
    file can vanish between rglob and the read."""
    return [path for path in sorted(root.rglob("*.txt")) if not path.name.startswith(".")]


def test_pinned_sentence_transformers_can_log_with_the_pinned_transformers():
    checked = 0
    for path in _requirement_files(REQ_ROOT):
        st_pins = _pins(path, "sentence-transformers")
        tf_pins = _pins(path, "transformers")
        for marker, st in st_pins.items():
            tf = tf_pins.get(marker)
            if tf is None:
                continue
            checked += 1
            assert _trains_with(tf, st), (
                f"{path.relative_to(REPO_ROOT)} pins sentence-transformers=={st} with "
                f"transformers=={tf} ({marker}); embedding training fails at its first "
                f"logging step with no attribute '_nested_gather'"
            )
    assert checked >= 4


def test_the_pair_that_broke_embedding_training_is_rejected():
    assert not _trains_with(Version("5.5.0"), Version("5.2.0"))
    assert _trains_with(Version("4.57.6"), Version("5.1.2"))


def test_a_filtered_copy_vanishing_mid_scan_is_not_read(tmp_path):
    (tmp_path / "single-env").mkdir()
    (tmp_path / "single-env" / "data-designer-deps.txt").write_text("x==1\n", encoding = "utf-8")
    # A dangling link reads like a copy deleted between rglob and read_text.
    (tmp_path / "single-env" / ".data-designer-deps-filtered-bpfrp0a_.txt").symlink_to(
        tmp_path / "gone.txt"
    )
    files = _requirement_files(tmp_path)
    assert [p.name for p in files] == ["data-designer-deps.txt"]
    for path in files:
        _pins(path, "transformers")

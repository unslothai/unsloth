# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seed path classification reads each Hub file name by string where pathlib would read it the same way: the
hand-split parts, suffixes, stems and compiled label matchers give exactly what PurePosixPath / Path / a fresh
re.search gave, for the POSIX and Windows flavours, on plain names and on the odd ones that fall back."""

import importlib.util
import random
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest


def _load(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("structlog")
    backend_root = Path(__file__).resolve().parent.parent
    monkeypatch.syspath_prepend(str(backend_root))
    spec = importlib.util.spec_from_file_location(
        "seed_path_split_under_test", backend_root / "routes" / "data_recipe" / "seed.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SEGMENTS = [
    "data",
    "train",
    "Train",
    "TEST",
    "validation",
    "dev",
    "__hidden",
    "__init__",
    ".git",
    ".",
    "..",
    "",
    "sub-train",
    "en",
    "sample-10BT",
    "x y",
    "a:b",
    "C:",
    "back\\slash",
    "İstanbul",
    "é",
    "-",
    "00",
]
_NAMES = [
    "train",
    "test",
    "train-00000-of-00004",
    "questions.train",
    "a.csv",
    ".hidden",
    "README",
    "x.",
    "y..",
    "..",
    ".",
    "TRAIN",
    "en-train",
    "000_00001",
    "__train",
    "tést",
    "dev-1",
    "a b",
    "c:d",
]
_EXTS = [
    ".parquet",
    ".PARQUET",
    ".json",
    ".jsonl",
    ".csv",
    ".json.gz",
    "",
    ".",
    ".parquet/",
    ".csv.parquet",
]


def _paths(count = 3000, seed = 0):
    rng = random.Random(seed)
    out = [
        "",
        ".",
        "..",
        "/",
        "//",
        "///",
        "a",
        "a/",
        "a/.",
        "a/..",
        "./a",
        "/a",
        "//a/b",
        "a//b",
        "a/./b",
        "C:/a.parquet",
        "C:a.parquet",
        "a\\b.parquet",
        "a/b.parquet ",
        "a/b .parquet",
        "a/.parquet",
        "a/b.",
        "a/b..parquet",
        "data/train-00000-of-00001.parquet",
        "İ.PARQUET",
        "data/x.Parquet",
    ]
    for _ in range(count):
        parts = [rng.choice(_SEGMENTS) for _ in range(rng.choice([0, 1, 1, 2, 3]))]
        path = "/".join(parts + [rng.choice(_NAMES) + rng.choice(_EXTS)])
        if rng.random() < 0.05:
            path = "/" + path
        if rng.random() < 0.05:
            path = path.replace("/", "//", 1)
        if rng.random() < 0.03:
            path += "/"
        out.append(path)
    return out


def test_marked_parts_and_loader_facts_match_pure_posix_path(monkeypatch):
    seed = _load(monkeypatch)
    for path in _paths():
        parsed = PurePosixPath(path)
        for prefix in (".", "__"):
            for which, parts in (("parts", parsed.parts), ("parent_parts", parsed.parent.parts)):
                expected = [p for p in parts if p.startswith(prefix) and set(p) != {"."}]
                assert seed._marked_parts(path, prefix, which) == expected, (path, prefix, which)
        assert seed._loader_path_facts(path)[2] == parsed.name, path


@pytest.mark.parametrize("flavour", [PurePosixPath, PureWindowsPath])
def test_name_parts_match_pathlib(monkeypatch, flavour):
    seed = _load(monkeypatch)
    monkeypatch.setattr(seed, "Path", flavour)
    for path in _paths():
        for value in (path, path.lower()):
            parsed = flavour(value)
            assert seed._name_parts(value) == (parsed.suffix, parsed.stem), (flavour, value)


def test_label_matcher_matches_a_fresh_search(monkeypatch):
    seed = _load(monkeypatch)
    labels = ["train", "dev", "en", "sample-10bt", "a.b", "x+y", "[", "", "validation"]
    for path in _paths(500):
        name = path.lower()
        for label in labels:
            expected = (
                re.search(rf"(?:^|{seed._SEP}){re.escape(label)}(?:$|{seed._SEP})", name)
                is not None
            )
            assert seed._label_in_name(name, label) == expected, (name, label)


@pytest.mark.parametrize("flavour", [PurePosixPath, PureWindowsPath])
def test_split_rank_and_suffix_filter_match_pathlib(monkeypatch, flavour):
    seed = _load(monkeypatch)
    monkeypatch.setattr(seed, "Path", flavour)
    paths = _paths(1500, seed = 1)
    for path in paths:
        lowered = path.lower()
        labels = seed._split_labels("train")
        if seed._split_folder(lowered, labels):
            expected = 0
        elif any(seed._label_in_name(flavour(lowered).stem, label) for label in labels):
            expected = 1
        else:
            expected = 2
        assert seed._split_rank(path, "train") == expected, (flavour, path)
    for suffix in (".parquet", ".json", ".JSONL", ".csv", ""):
        group = seed._builder_exts(suffix)
        expected = [p for p in paths if flavour(p).suffix.lower() in group]
        assert seed._of_suffix(paths, suffix) == expected, (flavour, suffix)

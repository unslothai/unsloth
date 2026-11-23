# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``_files_under_patterns`` parses each pattern and each matched file once per call, and still keeps exactly the
files, in the same order with the same duplicates, that the per-pair hidden / ignored checks kept."""

import importlib.util
import random
from pathlib import Path, PurePosixPath

import pytest


def _load(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("structlog")
    backend_root = Path(__file__).resolve().parent.parent
    monkeypatch.syspath_prepend(str(backend_root))
    spec = importlib.util.spec_from_file_location(
        "seed_glob_under_test", backend_root / "routes" / "data_recipe" / "seed.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _oracle(seed, patterns, data_files, as_the_loader_would):
    """The per (file, pattern) checks, kept verbatim."""
    ignored = seed._IGNORED_DATA_FILENAMES

    def marked(path, prefix, which):
        parsed = PurePosixPath(path)
        parts = parsed.parts if which == "parts" else parsed.parent.parts
        return [part for part in parts if part.startswith(prefix) and set(part) != {"."}]

    def ignored_by_the_loader(path, pattern):
        name = PurePosixPath(path).name
        return name in ignored and PurePosixPath(pattern).name != name

    def hidden_to_the_loader(path, pattern):
        for prefix, parts in ((".", "parts"), ("__", "parent_parts")):
            if len(marked(path, prefix, parts)) != len(marked(pattern, prefix, parts)):
                return True
        return False

    matchers = [(pattern, seed._glob_to_regex(pattern)) for pattern in patterns]
    return [
        f
        for f in data_files
        if any(
            m.match(f)
            and not (
                as_the_loader_would and (hidden_to_the_loader(f, p) or ignored_by_the_loader(f, p))
            )
            for p, m in matchers
        )
    ]


_DIRS = [
    "data",
    ".hidden",
    "__pycache__",
    "__x",
    "a.b",
    "..",
    ".",
    "sub",
    "train",
    "_y",
    "...",
    "x__",
]
_NAMES = [
    "train.parquet",
    "test-0001.jsonl",
    ".dot.csv",
    "__init__.json",
    "README.md",
    "config.json",
    "dataset_infos.json",
    "a[b].parquet",
    "x.parquet",
    "__train.json",
    "_meta.txt",
    "b.tar",
]
_PATTERN_DIRS = [
    "data",
    "*",
    "**",
    ".hidden",
    "__x",
    "__*",
    ".*",
    "sub",
    "[ds]*",
    "[!.]*",
    "a.b",
    "_y",
]
_PATTERN_NAMES = [
    "*.parquet",
    "*",
    "train*",
    ".*",
    "__*",
    "README.md",
    "config.json",
    "[a-z]*.json",
    "*[!q]",
    "a[b].parquet",
    "test-*.jsonl",
    "**",
]


def _path(rng, parts, names):
    depth = rng.randint(0, 3)
    return "/".join([rng.choice(parts) for _ in range(depth)] + [rng.choice(names)])


def _corpus(seed):
    rng = random.Random(seed)
    files = [_path(rng, _DIRS, _NAMES) for _ in range(rng.randint(0, 120))]
    files += rng.sample(files, k = min(len(files), rng.randint(0, 5)))  # duplicates
    rng.shuffle(files)
    patterns = [_path(rng, _PATTERN_DIRS, _PATTERN_NAMES) for _ in range(rng.randint(0, 5))]
    if files and rng.random() < 0.3:
        patterns.append(rng.choice(files))  # a pattern that names a file exactly
    return patterns, files


def test_randomised_corpora_match_the_per_pair_checks(monkeypatch):
    seed = _load(monkeypatch)
    differing = []
    kept_any = 0
    for case in range(1500):
        patterns, files = _corpus(case)
        for mode in (False, True):
            got = seed._files_under_patterns(patterns, files, as_the_loader_would = mode)
            want = _oracle(seed, patterns, files, mode)
            kept_any += bool(want)
            if got != want:
                differing.append((case, mode, patterns))
    assert not differing, differing[:3]
    assert kept_any > 500  # the corpora actually keep files, so the skip rules are exercised


@pytest.mark.parametrize(
    "patterns, files",
    [
        ([], ["data/train.parquet"]),
        (["data/*.parquet"], []),
        (
            ["**/*"],
            [
                "data/.hidden/x.parquet",
                "data/x.parquet",
                ".git/config.json",
                "__pycache__/a/x.parquet",
            ],
        ),
        (
            ["data/__x/*.json", "data/*/*.json"],
            ["data/__x/a.json", "data/__x/a.json", "data/y/a.json"],
        ),
        (["*.md", "README.md"], ["README.md", "x.md"]),
        (
            ["data/**/train.*", "data/train*"],
            ["data/a/b/train.json", "data/__train.json", "data/.train.json"],
        ),
        (["data/[!x]*.parquet", "data/x*.parquet"], ["data/x1.parquet", "data/y.parquet"]),
        (["./data/*.parquet"], ["./data/a.parquet", "data/a.parquet"]),
        (["../*"], ["../a", "a"]),
    ],
)
def test_edge_cases(monkeypatch, patterns, files):
    seed = _load(monkeypatch)
    for mode in (False, True):
        assert seed._files_under_patterns(patterns, files, as_the_loader_would = mode) == _oracle(
            seed, patterns, files, mode
        )


def test_each_file_and_pattern_is_parsed_once(monkeypatch):
    seed = _load(monkeypatch)
    seen = []
    real = seed._marked_parts

    def spy(path, prefix, which):
        seen.append(path)
        return real(path, prefix, which)

    monkeypatch.setattr(seed, "_marked_parts", spy)
    files = [f"data/train-{i:05d}.parquet" for i in range(200)]
    patterns = ["data/train-*.parquet", "data/*.parquet", "data/train-0*.parquet"]
    kept = seed._files_under_patterns(patterns, files, as_the_loader_would = True)
    assert kept == files
    # two facts (hidden parts, dunder parents) per pattern and per matched file, never per pair
    assert len(seen) == 2 * (len(patterns) + len(files))

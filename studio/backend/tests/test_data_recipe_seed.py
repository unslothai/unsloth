# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException


def _seed_route_source() -> str:
    return (
        Path(__file__).resolve().parent.parent / "routes" / "data_recipe" / "seed.py"
    ).read_text(encoding = "utf-8")


def test_seed_inspect_load_kwargs_disables_remote_code_execution():
    assert '"trust_remote_code": False' in _seed_route_source()


class _FakeUpload:
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._content = content

    async def read(self) -> bytes:
        return self._content


def _load_seed_route(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    inline_extraction = True,
):
    pytest.importorskip("fastapi")
    pytest.importorskip("multipart")
    pytest.importorskip("structlog")

    backend_root = Path(__file__).resolve().parent.parent
    monkeypatch.syspath_prepend(str(backend_root))
    route_path = backend_root / "routes" / "data_recipe" / "seed.py"
    spec = importlib.util.spec_from_file_location("seed_under_test", route_path)
    assert spec is not None and spec.loader is not None
    seed_route = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seed_route)
    seed_route.UNSTRUCTURED_UPLOAD_ROOT = tmp_path / "unstructured-uploads"
    if inline_extraction:
        # Unit cases inject extractor failures in this process. Process isolation has separate tests.
        async def extract(file_path, ext):
            return seed_route._extract_text_from_file(file_path, ext)

        monkeypatch.setattr(seed_route, "_extract_text_from_file_async", extract)
    return seed_route


def _run_upload(
    seed_route,
    filename: str,
    content: bytes,
    block_id: str = "block",
):
    return asyncio.run(
        seed_route.upload_unstructured_file(_FakeUpload(filename, content), block_id)
    )


def _block_files(seed_route, block_id: str = "block") -> list[str]:
    block_dir = seed_route.UNSTRUCTURED_UPLOAD_ROOT / block_id
    if not block_dir.exists():
        return []
    return sorted(path.name for path in block_dir.iterdir())


def _raise(exc: BaseException):
    def raise_exc(*args, **kwargs):
        raise exc

    return raise_exc


@pytest.mark.parametrize(
    ("filename", "package"),
    [
        ("paper.pdf", "pymupdf4llm"),
        ("notes.docx", "mammoth"),
    ],
)
def test_unstructured_upload_names_missing_extractor_dependency(
    monkeypatch, tmp_path, filename, package
):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    monkeypatch.setattr(
        seed_route,
        "_extract_text_from_file",
        _raise(ModuleNotFoundError(f"No module named {package!r}", name = package)),
    )

    result = _run_upload(seed_route, filename, b"%PDF-1.7")

    assert result.status == "error"
    assert (
        result.error
        == f"Cannot read {Path(filename).suffix} files: the '{package}' package is not installed."
    )
    assert _block_files(seed_route) == []


def test_unstructured_upload_keeps_txt_path_working(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)

    result = _run_upload(seed_route, "notes.txt", b"hello")

    assert result.status == "ok"
    assert result.error is None
    assert any(name.endswith(".txt") for name in _block_files(seed_route))
    assert any(name.endswith(".extracted.txt") for name in _block_files(seed_route))


@pytest.mark.parametrize(
    "exc",
    [
        ImportError("cannot import internal symbol"),
        ModuleNotFoundError(
            "No module named 'missing_transitive_pkg'",
            name = "missing_transitive_pkg",
        ),
    ],
)
def test_unstructured_upload_import_errors_stay_generic(monkeypatch, tmp_path, exc):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    monkeypatch.setattr(seed_route, "_extract_text_from_file", _raise(exc))
    result = _run_upload(seed_route, "paper.pdf", b"%PDF-1.7")

    assert result.status == "error"
    assert result.error == "Text extraction failed."
    assert _block_files(seed_route) == []


_TEST_UPLOAD_UID = "0f" * 16


def test_remove_unstructured_block_deletes_directory(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    _run_upload(seed_route, "notes.txt", b"hello", block_id = _TEST_UPLOAD_UID)
    assert _block_files(seed_route, _TEST_UPLOAD_UID) != []

    result = asyncio.run(seed_route.remove_unstructured_block(_TEST_UPLOAD_UID))

    assert result == {"status": "ok", "deleted": True}
    assert not (seed_route.UNSTRUCTURED_UPLOAD_ROOT / _TEST_UPLOAD_UID).exists()


def test_remove_unstructured_block_missing_directory_is_ok(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)

    result = asyncio.run(seed_route.remove_unstructured_block(_TEST_UPLOAD_UID))

    assert result == {"status": "ok", "deleted": False}


def test_remove_unstructured_block_rejects_unsafe_ids(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)

    with pytest.raises(seed_route.HTTPException) as exc:
        asyncio.run(seed_route.remove_unstructured_block("../escape"))

    assert exc.value.status_code == 400


def test_remove_unstructured_block_rejects_legacy_node_ids(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    _run_upload(seed_route, "notes.txt", b"hello", block_id = "n1")
    assert _block_files(seed_route, "n1") != []

    with pytest.raises(seed_route.HTTPException) as exc:
        asyncio.run(seed_route.remove_unstructured_block("n1"))

    assert exc.value.status_code == 400
    assert _block_files(seed_route, "n1") != []


def test_remove_unstructured_block_rejects_symlink_escape(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "victim.txt").write_text("keep me")
    root = seed_route.UNSTRUCTURED_UPLOAD_ROOT
    root.mkdir(parents = True)
    (root / _TEST_UPLOAD_UID).symlink_to(outside)

    with pytest.raises(seed_route.HTTPException) as exc:
        asyncio.run(seed_route.remove_unstructured_block(_TEST_UPLOAD_UID))

    assert exc.value.status_code == 400
    assert (outside / "victim.txt").exists()


def test_remove_unstructured_block_fails_if_directory_remains(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    root = seed_route.UNSTRUCTURED_UPLOAD_ROOT
    block_dir = root / _TEST_UPLOAD_UID
    block_dir.mkdir(parents = True)
    (block_dir / "victim.txt").write_text("keep me")

    calls = []

    def noop_rmtree(path, *args, **kwargs):
        calls.append((path, args, kwargs))

    monkeypatch.setattr(seed_route.shutil, "rmtree", noop_rmtree)

    with pytest.raises(seed_route.HTTPException) as exc:
        asyncio.run(seed_route.remove_unstructured_block(_TEST_UPLOAD_UID))

    assert calls
    assert exc.value.status_code == 500
    assert block_dir.exists()


def test_total_upload_quota_is_scoped_per_block(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    monkeypatch.setattr(seed_route, "UNSTRUCTURED_RECIPE_UPLOAD_TOTAL_MAX_BYTES", 10)

    first = _run_upload(seed_route, "a.txt", b"123456789")
    assert first.status == "ok"

    with pytest.raises(seed_route.HTTPException) as exc:
        _run_upload(seed_route, "b.txt", b"123")
    assert exc.value.status_code == 413

    # Another block starts with its own untouched budget.
    other = _run_upload(seed_route, "c.txt", b"123", block_id = "other")
    assert other.status == "ok"


# A desktop drop names a local file of any size, so the cap has to be enforced
# on its stat. Reading first let a multi-gigabyte drop into backend memory
# before the 413 (#9036).
def test_an_oversized_native_drop_is_refused_before_it_is_read(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    huge = tmp_path / "corpus.txt"
    huge.write_bytes(b"x" * 64)

    reads: list[str] = []
    real_open = Path.open

    def tracking_open(self, *args, **kwargs):
        reads.append(self.name)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)
    monkeypatch.setattr(seed_route, "UNSTRUCTURED_RECIPE_UPLOAD_MAX_BYTES", 32)
    monkeypatch.setattr(
        seed_route,
        "verify_native_path_lease",
        lambda *a, **k: SimpleNamespace(canonical_path = huge),
        raising = False,
    )
    monkeypatch.setitem(
        sys.modules,
        "utils.native_path_leases",
        SimpleNamespace(
            NativePathLeaseError = RuntimeError,
            verify_native_path_lease = lambda *a, **k: SimpleNamespace(canonical_path = huge),
        ),
    )

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            seed_route.upload_unstructured_file(None, "block", native_path_lease = "signed-lease")
        )
    assert excinfo.value.status_code == 413
    assert reads == [], "the file was opened before the size check"


# The block's remaining budget bounds the read too, so a file that grew between
# the stat and the read cannot slip past it.
def test_a_native_drop_over_the_block_budget_is_refused(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    dropped = tmp_path / "notes.txt"
    dropped.write_bytes(b"y" * 64)

    monkeypatch.setattr(seed_route, "UNSTRUCTURED_RECIPE_UPLOAD_TOTAL_MAX_BYTES", 16)
    monkeypatch.setitem(
        sys.modules,
        "utils.native_path_leases",
        SimpleNamespace(
            NativePathLeaseError = RuntimeError,
            verify_native_path_lease = lambda *a, **k: SimpleNamespace(canonical_path = dropped),
        ),
    )

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            seed_route.upload_unstructured_file(None, "block", native_path_lease = "signed-lease")
        )
    assert excinfo.value.status_code == 413


class _BlockPlugin:
    """Meta path finder making the optional seed plugin look uninstalled."""

    def __init__(self, name: str = "data_designer_unstructured_seed"):
        self.name = name
        self.attempts = 0

    def find_spec(
        self,
        fullname,
        path = None,
        target = None,
    ):
        if fullname == self.name or fullname.startswith(self.name + "."):
            self.attempts += 1
            raise ModuleNotFoundError(f"No module named {fullname!r}", name = fullname)
        return None


def _without_plugin(monkeypatch, seed_route):
    import sys

    blocker = _BlockPlugin()
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
    for name in [m for m in sys.modules if m.split(".")[0] == blocker.name]:
        monkeypatch.delitem(sys.modules, name)
    seed_route._CHUNKING = None
    return blocker


def test_unstructured_preview_reports_unavailable_without_the_plugin(monkeypatch, tmp_path):
    """Deferring the plugin import must not change what a missing plugin looks like."""
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    _without_plugin(monkeypatch, seed_route)

    assert seed_route._chunking() is None

    with pytest.raises(seed_route.HTTPException) as exc:
        seed_route._read_preview_rows_from_unstructured_file(
            path = tmp_path / "a.txt", preview_size = 5, chunk_size = None, chunk_overlap = None
        )
    assert exc.value.status_code == 500
    assert "Unstructured seed support not available" in exc.value.detail

    with pytest.raises(seed_route.HTTPException) as exc:
        seed_route._read_preview_rows_from_multi_files(
            block_id = "block",
            file_ids = ["a"],
            file_names = ["a.txt"],
            preview_size = 5,
            chunk_size = None,
            chunk_overlap = None,
        )
    assert exc.value.status_code == 500
    assert "Unstructured seed support not available" in exc.value.detail


def test_missing_plugin_is_probed_once(monkeypatch, tmp_path):
    """A failed probe is remembered, so previews do not retry the import every time."""
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    blocker = _without_plugin(monkeypatch, seed_route)

    assert seed_route._chunking() is None
    assert seed_route._chunking() is None
    assert blocker.attempts == 1


def test_text_extraction_falls_back_to_raw_without_the_plugin(monkeypatch, tmp_path):
    """normalize_unstructured_text lives in the plugin; without it raw text stands."""
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    _without_plugin(monkeypatch, seed_route)
    source = tmp_path / "notes.txt"
    source.write_text("a\n\n\n\nb", encoding = "utf-8")

    # The plugin is what collapses the run of blank lines.
    assert seed_route._extract_text_from_file(source, ".txt") == "a\n\n\n\nb"


def test_plugin_resolution_survives_a_reload_and_normalizes(monkeypatch, tmp_path):
    """With the plugin installed the same call sites still go through it."""
    pytest.importorskip("data_designer_unstructured_seed")
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    seed_route._CHUNKING = None

    chunking = seed_route._chunking()
    assert chunking is not None
    assert chunking.resolve_chunking(0, 0)[0] == 1
    source = tmp_path / "notes.txt"
    source.write_text("a\n\n\n\nb", encoding = "utf-8")
    assert seed_route._extract_text_from_file(source, ".txt") == "a\n\nb"


def test_a_backend_executed_seed_resolves_the_endpoint_on_the_backend(monkeypatch):
    """The seed is fetched in THIS process, so the endpoint must be ours.

    A remote browser is told the public default for a loopback mirror (it cannot
    reach the backend's localhost), so letting the client's value through would
    bypass the mirror on exactly the deployments that need it. A value the user
    typed into the seed node is still honoured.
    """
    pytest.importorskip("fastapi")
    backend_root = Path(__file__).resolve().parent.parent
    monkeypatch.syspath_prepend(str(backend_root))
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9700")

    from routes.data_recipe.jobs import _resolve_seed_endpoint

    recipe = {"seed_config": {"source": {"seed_type": "hf", "path": "a/b", "endpoint": None}}}
    _resolve_seed_endpoint(recipe)
    assert recipe["seed_config"]["source"]["endpoint"] == "http://127.0.0.1:9700"

    recipe = {"seed_config": {"source": {"seed_type": "hf", "path": "a/b"}}}
    _resolve_seed_endpoint(recipe)
    assert recipe["seed_config"]["source"]["endpoint"] == "http://127.0.0.1:9700"

    explicit = {
        "seed_config": {
            "source": {"seed_type": "hf", "path": "a/b", "endpoint": "https://hub.internal"}
        }
    }
    _resolve_seed_endpoint(explicit)
    assert explicit["seed_config"]["source"]["endpoint"] == "https://hub.internal"

    # Nothing to resolve for the other seed types, and no crash on a malformed recipe.
    other = {"seed_config": {"source": {"seed_type": "local", "paths": []}}}
    _resolve_seed_endpoint(other)
    assert "endpoint" not in other["seed_config"]["source"]
    _resolve_seed_endpoint({})
    _resolve_seed_endpoint({"seed_config": "nope"})


_GSM8K_FILES = [
    "main/test-00000-of-00001.parquet",
    "main/train-00000-of-00001.parquet",
    "socratic/test-00000-of-00001.parquet",
    "socratic/train-00000-of-00001.parquet",
]


@pytest.mark.parametrize(
    ("files", "split", "subset", "expected"),
    [
        # No card: the loader reads every train-named file as one split.
        (_GSM8K_FILES, "train", None, "datasets/org/repo/**/train-*.parquet"),
        (_GSM8K_FILES, "test", "main", "datasets/org/repo/main/test-*.parquet"),
        (_GSM8K_FILES, "train", "socratic", "datasets/org/repo/socratic/train-*.parquet"),
        (_GSM8K_FILES, "train", "default", "datasets/org/repo/**/train-*.parquet"),
        (
            ["data/test-00000-of-00001.parquet", "data/train-00000-of-00002.parquet"],
            "train",
            "default",
            "datasets/org/repo/data/train-*.parquet",
        ),
        (["test.csv", "train.csv"], "train", None, "datasets/org/repo/train*.csv"),
        (["Train_0.jsonl", "Test_0.jsonl"], "train", None, "datasets/org/repo/Train_*.jsonl"),
        (
            ["raw/gsm_test.jsonl", "raw/gsm_train.jsonl"],
            "train",
            None,
            "datasets/org/repo/raw/gsm_train*.jsonl",
        ),
        # A split named by the file itself can still be sharded, so the pattern has
        # to reach the siblings instead of pinning the first shard.
        (
            ["train.jsonl", "train_2.jsonl", "test.jsonl"],
            "train",
            None,
            "datasets/org/repo/train*.jsonl",
        ),
        (
            ["raw/gsm_test.jsonl", "raw/gsm_train.jsonl", "raw/gsm_train_2.jsonl"],
            "train",
            None,
            "datasets/org/repo/raw/gsm_train*.jsonl",
        ),
        (
            ["en/test/0000.parquet", "en/train/0000.parquet", "fr/train/0000.parquet"],
            "train",
            "fr",
            "datasets/org/repo/fr/train/**/*.parquet",
        ),
        (["data/part-0.parquet"], "train", "main", "datasets/org/repo/data/**/*.parquet"),
    ],
)
def test_seed_hf_path_keeps_the_split_and_subset(
    monkeypatch, tmp_path, files, split, subset, expected
):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    assert seed_route._resolve_seed_hf_path("org/repo", files, split, subset) == expected


def test_seed_preview_file_comes_from_the_chosen_subset(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    assert (
        seed_route._select_best_file(_GSM8K_FILES, "train", "socratic")
        == "socratic/train-00000-of-00001.parquet"
    )


# fineweb-edu: the config name is not a folder, so a folder-name guess reads a
# different config entirely.
_SAMPLE_FILES = [
    "data/CC-MAIN-2013-20/train-00000-of-00014.parquet",
    "sample/10BT/000_00000.parquet",
    "sample/10BT/001_00000.parquet",
]
_SAMPLE_CONFIGS = [
    {"config_name": "default", "data_files": [{"split": "train", "path": "data/*/*"}]},
    {"config_name": "sample-10BT", "data_files": [{"split": "train", "path": "sample/10BT/*"}]},
]


@pytest.mark.parametrize(
    ("subset", "expected"),
    [
        ("sample-10BT", "datasets/org/repo/sample/10BT/*.parquet"),
        (None, "datasets/org/repo/data/*/*.parquet"),
    ],
)
def test_seed_hf_path_follows_the_dataset_card_config_mapping(
    monkeypatch, tmp_path, subset, expected
):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    resolved = seed_route._resolve_seed_hf_path(
        "org/repo", _SAMPLE_FILES, "train", subset, _SAMPLE_CONFIGS
    )
    assert resolved == expected


def test_seed_hf_path_ignores_a_card_mapping_pointing_at_nothing(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    configs = [{"config_name": "main", "data_files": [{"split": "train", "path": "gone/*"}]}]
    resolved = seed_route._resolve_seed_hf_path("org/repo", _GSM8K_FILES, "train", "main", configs)
    assert resolved == "datasets/org/repo/main/train-*.parquet"


@pytest.mark.parametrize(
    "data_files",
    [
        "mapped/train-00000.parquet",
        ["mapped/train-00000.parquet"],
        {"train": "mapped/train-*.parquet"},
        [{"split": "train", "path": "mapped/train-*.parquet"}],
    ],
)
def test_seed_hf_path_reads_every_card_data_files_shape(monkeypatch, tmp_path, data_files):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["mapped/train-00000.parquet", "other/train-00000.parquet"]
    configs = [{"config_name": "unrelated", "data_files": data_files}]
    resolved = seed_route._resolve_seed_hf_path("org/repo", files, "train", "unrelated", configs)
    assert resolved.startswith("datasets/org/repo/mapped/")


@pytest.mark.parametrize(
    ("configs", "expected"),
    [
        # A card with one config uses it whatever it is called.
        (
            [
                {
                    "config_name": "plain_text",
                    "data_files": [{"split": "train", "path": "pt/train-*"}],
                }
            ],
            "datasets/org/repo/pt/train-*.parquet",
        ),
        # Another config can be flagged as the default one.
        (
            [
                {"config_name": "a", "data_files": [{"split": "train", "path": "a/train-*"}]},
                {
                    "config_name": "b",
                    "default": True,
                    "data_files": [{"split": "train", "path": "pt/train-*"}],
                },
            ],
            "datasets/org/repo/pt/train-*.parquet",
        ),
    ],
)
def test_seed_hf_path_finds_the_default_config_without_a_subset(
    monkeypatch, tmp_path, configs, expected
):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["pt/train-0.parquet", "a/train-0.parquet"]
    assert seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs) == expected


def test_seed_card_globs_keep_their_character_classes(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["data/train-2-of-9.parquet", "data/train-7-of-9.parquet"]
    assert seed_route._files_under_patterns(["data/train-[0-4]*.parquet"], files) == [
        "data/train-2-of-9.parquet"
    ]
    assert seed_route._files_under_patterns(["data/[!x]*.parquet"], files) == files
    # An unclosed bracket is a literal, not a syntax error.
    assert seed_route._files_under_patterns(["a[b.parquet"], ["a[b.parquet"]) == ["a[b.parquet"]


def test_seed_hf_path_covers_a_split_declared_across_two_folders(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["sets/a/train-0.parquet", "sets/b/train-0.parquet", "other/test-0.parquet"]
    configs = [
        {
            "config_name": "default",
            "data_files": [
                {"split": "train", "path": ["sets/a/train-*.parquet", "sets/b/train-*.parquet"]}
            ],
        }
    ]
    # One glob cannot name two folders, so cover the folder holding both rather
    # than dropping one of them, keeping the split in the pattern.
    assert (
        seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs)
        == "datasets/org/repo/sets/**/*train*.parquet"
    )


_LABELLED_FILES = [
    "main-train.parquet",
    "main-test.parquet",
    "socratic-train.parquet",
    "socratic-test.parquet",
]


@pytest.mark.parametrize(
    ("subset", "expected"),
    [
        ("socratic", "datasets/org/repo/socratic-train*.parquet"),
        ("main", "datasets/org/repo/main-train*.parquet"),
    ],
)
def test_seed_hf_path_reads_a_subset_labelled_in_the_file_name(
    monkeypatch, tmp_path, subset, expected
):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    resolved = seed_route._resolve_seed_hf_path("org/repo", _LABELLED_FILES, "train", subset)
    assert resolved == expected


@pytest.mark.parametrize(
    ("files", "split", "expected"),
    [
        (["data/dev.jsonl", "data/train.jsonl"], "validation", "datasets/org/repo/data/dev*.jsonl"),
        (
            ["data/valid-0.parquet", "data/train-0.parquet"],
            "validation",
            "datasets/org/repo/data/valid-*.parquet",
        ),
        (
            ["data/training-0.parquet", "data/test-0.parquet"],
            "train",
            "datasets/org/repo/data/*train*.parquet",
        ),
    ],
)
def test_seed_hf_path_follows_the_split_aliases_datasets_uses(
    monkeypatch, tmp_path, files, split, expected
):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    assert seed_route._resolve_seed_hf_path("org/repo", files, split) == expected


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        # A folder qualified by a separator is the split folder too.
        (["train_a/0.parquet", "test/0.parquet"], "datasets/org/repo/train_a/**/*.parquet"),
        (
            ["data/train/0.parquet", "data/test/0.parquet"],
            "datasets/org/repo/data/train/**/*.parquet",
        ),
        # "pretrain" is a different word, so train/ still wins.
        (["pretrain/0.parquet", "train/0.parquet"], "datasets/org/repo/train/**/*.parquet"),
    ],
)
def test_seed_hf_path_reads_a_qualified_split_folder(monkeypatch, tmp_path, files, expected):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    assert seed_route._resolve_seed_hf_path("org/repo", files, "train") == expected


def test_seed_hf_path_widens_through_an_alias_named_split(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["sets/a/dev-0.parquet", "sets/b/dev-0.parquet", "sets/a/train-0.parquet"]
    configs = [
        {
            "config_name": "default",
            "data_files": [
                {"split": "validation", "path": ["sets/a/dev-*.parquet", "sets/b/dev-*.parquet"]},
                {"split": "train", "path": "sets/a/train-*.parquet"},
            ],
        }
    ]
    # The files say dev, not validation, so the widened form has to say dev too
    # or it takes the train shard with it.
    assert (
        seed_route._resolve_seed_hf_path("org/repo", files, "validation", None, configs)
        == "datasets/org/repo/sets/**/*dev*.parquet"
    )


def test_seed_hf_path_counts_a_digit_as_a_label_separator(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    # The loader's own separators include digits, so train1 is train.
    files = ["data/train-0.parquet", "data/train1.parquet", "data/test-0.parquet"]
    assert (
        seed_route._resolve_seed_hf_path("org/repo", files, "train")
        == "datasets/org/repo/data/*train*.parquet"
    )
    assert (
        seed_route._resolve_seed_hf_path(
            "org/repo", ["data/train-0.parquet", "data/testing-0.parquet"], "train"
        )
        == "datasets/org/repo/data/train-*.parquet"
    )


def test_seed_hf_path_keeps_a_split_spread_over_sibling_folders(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["sets/a/part.parquet", "sets/b/part.parquet", "sets/c/test.parquet"]
    configs = [
        {
            "config_name": "default",
            "data_files": [
                {"split": "train", "path": ["sets/a/part.parquet", "sets/b/part.parquet"]},
                {"split": "test", "path": "sets/c/test.parquet"},
            ],
        }
    ]
    # Nothing names the split, but the two declared files share a name the test
    # file does not, so the union stays off sets/c.
    assert (
        seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs)
        == "datasets/org/repo/sets/**/part*.parquet"
    )


def test_seed_hf_path_unions_declared_files_with_unrelated_names(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["sets/a/part-a.parquet", "sets/b/chunk-b.parquet", "sets/c/test.parquet"]
    configs = [
        {
            "config_name": "default",
            "data_files": [
                {"split": "train", "path": ["sets/a/part-a.parquet", "sets/b/chunk-b.parquet"]},
                {"split": "test", "path": "sets/c/test.parquet"},
            ],
        }
    ]
    # Nothing is shared but the first letters, and a class is the only union the
    # reader understands: it rejects {a,b} outright.
    assert (
        seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs)
        == "datasets/org/repo/sets/**/[cp]*.parquet"
    )


def test_seed_format_inference_stops_where_the_loader_stops(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    # 200 csv sort ahead of 201 parquet, and the loader only looks at the first
    # 200, so it builds csv and the recipe has to agree.
    files = [f"data/{i:03d}.csv" for i in range(200)] + [
        f"data/z{i:03d}.parquet" for i in range(201)
    ]
    configs = [{"config_name": "default", "data_files": [{"split": "train", "path": "data/*"}]}]
    assert (
        seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs)
        == "datasets/org/repo/data/*.csv"
    )


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        # Most files wins, so the long csv name does not decide the format.
        (
            ["data/a-very-long-name.csv", "data/z.parquet", "data/y.parquet"],
            "datasets/org/repo/data/*.parquet",
        ),
        (["data/a.csv", "data/z.parquet"], "datasets/org/repo/data/*.parquet"),
        (["data/a.csv", "data/b.csv"], "datasets/org/repo/data/*.csv"),
    ],
)
def test_seed_hf_path_picks_the_format_the_loader_would_build(
    monkeypatch, tmp_path, files, expected
):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    configs = [{"config_name": "default", "data_files": [{"split": "train", "path": "data/*"}]}]
    assert seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs) == expected


def test_seed_hf_path_combines_bare_and_explicit_train_entries(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    configs = [
        {
            "config_name": "default",
            "data_files": [
                "data/part-a.parquet",
                {"split": "train", "path": "data/part-b.parquet"},
            ],
        }
    ]
    assert seed_route._declared_split_patterns(configs, "train") == [
        "data/part-a.parquet",
        "data/part-b.parquet",
    ]
    files = ["data/part-a.parquet", "data/part-b.parquet", "data/test-0.parquet"]
    # Neither declared file names the split, but they share a name the test file
    # does not, so the union stays exact.
    assert (
        seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs)
        == "datasets/org/repo/data/part-*.parquet"
    )


def test_seed_hf_path_keeps_a_folder_subset_whose_files_are_generic(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    # en/ is the config even though its file says nothing about the split, so
    # other/train.parquet must not win it.
    files = ["en/data.parquet", "other/train.parquet"]
    assert (
        seed_route._resolve_seed_hf_path("org/repo", files, "train", "en")
        == "datasets/org/repo/en/**/*.parquet"
    )


@pytest.mark.parametrize(
    ("subset", "expected"),
    [
        ("main", "datasets/org/repo/train-main*.parquet"),
        ("socratic", "datasets/org/repo/train-socratic*.parquet"),
    ],
)
def test_seed_hf_path_keeps_a_subset_written_after_the_split(
    monkeypatch, tmp_path, subset, expected
):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["train-main.parquet", "test-main.parquet", "train-socratic.parquet"]
    # train-*.parquet fits the subset slice but takes the other subset with it,
    # so the candidate has to be judged against the whole listing.
    assert seed_route._resolve_seed_hf_path("org/repo", files, "train", subset) == expected


@pytest.mark.parametrize(
    ("subset", "expected"),
    [
        ("french", "datasets/org/repo/fr/train*.parquet"),
        ("english", "datasets/org/repo/en/train*.parquet"),
    ],
)
def test_seed_hf_path_scopes_a_config_to_its_data_dir(monkeypatch, tmp_path, subset, expected):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    configs = [
        {"config_name": "english", "data_dir": "en"},
        {"config_name": "french", "data_dir": "fr"},
    ]
    files = ["en/train.parquet", "en/test.parquet", "fr/train.parquet", "fr/test.parquet"]
    assert seed_route._resolve_seed_hf_path("org/repo", files, "train", subset, configs) == expected


def test_seed_hf_path_reads_data_files_relative_to_data_dir(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    configs = [
        {
            "config_name": "a",
            "data_dir": "d",
            "data_files": [{"split": "train", "path": "train-*.parquet"}],
        }
    ]
    files = ["d/train-0.parquet", "d/test-0.parquet"]
    assert (
        seed_route._resolve_seed_hf_path("org/repo", files, "train", "a", configs)
        == "datasets/org/repo/d/train-*.parquet"
    )


def test_seed_format_vote_ignores_folder_metadata(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    configs = [{"config_name": "default", "data_files": [{"split": "train", "path": "data/*"}]}]
    files = ["data/metadata.csv", "data/metadata2.csv", "data/shard.parquet"]
    # metadata.csv never decides a builder for the loader, so it cannot here.
    assert (
        seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs)
        == "datasets/org/repo/data/*.parquet"
    )


@pytest.mark.parametrize(
    ("subset", "expected"),
    [
        ("Foo", "datasets/org/repo/F/train.parquet"),
        ("foo", "datasets/org/repo/f/train.parquet"),
    ],
)
def test_seed_hf_path_matches_config_names_case_sensitively(
    monkeypatch, tmp_path, subset, expected
):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    configs = [
        {"config_name": "Foo", "data_files": [{"split": "train", "path": "F/train.parquet"}]},
        {"config_name": "foo", "data_files": [{"split": "train", "path": "f/train.parquet"}]},
    ]
    files = ["F/train.parquet", "f/train.parquet"]
    assert seed_route._resolve_seed_hf_path("org/repo", files, "train", subset, configs) == expected


def test_seed_hf_path_keeps_a_hand_listed_split_off_its_neighbours(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["data/apple.parquet", "data/banana.parquet", "data/avocado.parquet"]
    configs = [
        {
            "config_name": "default",
            "data_files": [
                {"split": "train", "path": ["data/apple.parquet", "data/banana.parquet"]},
                {"split": "test", "path": "data/avocado.parquet"},
            ],
        }
    ]

    resolved = seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs)

    matched = seed_route._files_under_patterns([resolved[len("datasets/org/repo/") :]], files)
    assert sorted(matched) == ["data/apple.parquet", "data/banana.parquet"]


def test_seed_hf_path_never_widens_a_declared_split_over_another(monkeypatch, tmp_path):
    """No glob names these two alone, so the recipe reads what it can name."""
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["data/ax.parquet", "data/bx.parquet", "data/cx.parquet"]
    configs = [
        {
            "config_name": "default",
            "data_files": [
                {"split": "train", "path": ["data/ax.parquet", "data/cx.parquet"]},
                {"split": "test", "path": "data/bx.parquet"},
            ],
        }
    ]

    resolved = seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs)

    matched = seed_route._files_under_patterns([resolved[len("datasets/org/repo/") :]], files)
    assert sorted(matched) == ["data/ax.parquet", "data/cx.parquet"]


def test_seed_hf_path_keeps_shards_whose_folders_also_hold_other_splits(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = [
        "a/train-0.parquet",
        "a/test-0.parquet",
        "b/train-1.parquet",
        "b/test-1.parquet",
    ]

    resolved = seed_route._resolve_seed_hf_path("org/repo", files, "train")

    matched = seed_route._files_under_patterns([resolved[len("datasets/org/repo/") :]], files)
    assert sorted(matched) == ["a/train-0.parquet", "b/train-1.parquet"]


def test_seed_hf_path_refuses_a_mixed_json_split_it_cannot_name(monkeypatch, tmp_path):
    """*.json* is run over the whole repo, so a a.json.gz would be read too."""
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["data/a.json", "data/b.jsonl"]
    configs = [{"config_name": "default", "data_files": [{"split": "train", "path": "data/*"}]}]

    resolved = seed_route._resolve_seed_hf_path(
        "org/repo", files, "train", None, configs, [*files, "data/a.json.gz"]
    )

    assert resolved is None


def test_seed_hf_path_keeps_a_trailing_globstar_a_whole_component(monkeypatch, tmp_path):
    """fsspec refuses data/**.parquet outright, so the card's data/** grows a /*."""
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["data/a/train.parquet", "data/train.parquet"]
    configs = [{"config_name": "default", "data_files": [{"split": "train", "path": "data/**"}]}]

    resolved = seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs)

    assert resolved == "datasets/org/repo/data/**/*.parquet"


def test_seed_hf_path_leaves_hidden_folders_out_of_a_broad_card_glob(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["data/train.parquet", ".backup/train.parquet"]
    configs = [{"config_name": "default", "data_files": [{"split": "train", "path": "**/*"}]}]

    resolved = seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs)

    matched = seed_route._files_under_patterns([resolved[len("datasets/org/repo/") :]], files)
    assert matched == ["data/train.parquet"]


def test_seed_hf_path_reads_a_hidden_folder_the_card_names(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["data/train.parquet", ".backup/train.parquet"]
    configs = [{"config_name": "default", "data_files": [{"split": "train", "path": ".backup/*"}]}]

    resolved = seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs)

    matched = seed_route._files_under_patterns([resolved[len("datasets/org/repo/") :]], files)
    assert matched == [".backup/train.parquet"]


def test_seed_hf_path_keeps_both_extensions_of_one_builder(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["data/a.json", "data/b.jsonl"]
    configs = [{"config_name": "default", "data_files": [{"split": "train", "path": "data/*"}]}]

    resolved = seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs)

    assert resolved == "datasets/org/repo/data/*.json*"


def test_seed_hf_path_finds_sharded_names_under_a_config_data_dir(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = [
        "scope/data/train-00000-of-00001.parquet",
        "scope/data/test-00000-of-00001.parquet",
        "scope/extras/train-extra.parquet",
    ]
    configs = [{"config_name": "scoped", "data_dir": "scope"}]

    resolved = seed_route._resolve_seed_hf_path("org/repo", files, "train", "scoped", configs)

    matched = seed_route._files_under_patterns([resolved[len("datasets/org/repo/") :]], files)
    assert matched == ["scope/data/train-00000-of-00001.parquet"]


def test_seed_hf_path_refuses_a_declared_split_no_glob_can_name(monkeypatch, tmp_path):
    """aa and bb cannot be told from ab and ba by any class a glob can carry."""
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["data/aa.parquet", "data/ab.parquet", "data/ba.parquet", "data/bb.parquet"]
    configs = [
        {
            "config_name": "default",
            "data_files": [
                {"split": "train", "path": ["data/aa.parquet", "data/bb.parquet"]},
                {"split": "test", "path": ["data/ab.parquet", "data/ba.parquet"]},
            ],
        }
    ]

    assert seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs) is None


def test_seed_hf_path_stops_at_the_sharded_names_the_loader_reads_first(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = [
        "data/train-00000-of-00001.parquet",
        "data/test-00000-of-00001.parquet",
        "extras/train-extra.parquet",
    ]

    resolved = seed_route._resolve_seed_hf_path("org/repo", files, "train")

    matched = seed_route._files_under_patterns([resolved[len("datasets/org/repo/") :]], files)
    assert matched == ["data/train-00000-of-00001.parquet"]


@pytest.mark.parametrize(
    "configs",
    [
        [
            {
                "config_name": "default",
                "data_dir": "./data",
                "data_files": [
                    {"split": "train", "path": "a.parquet"},
                    {"split": "test", "path": "b.parquet"},
                ],
            }
        ],
        [
            {
                "config_name": "default",
                "data_dir": ".",
                "data_files": [
                    {"split": "train", "path": "./data/a.parquet"},
                    {"split": "test", "path": "./data/b.parquet"},
                ],
            }
        ],
    ],
)
def test_seed_hf_path_reads_a_card_that_spells_paths_with_a_dot(monkeypatch, tmp_path, configs):
    """The listing has no ./ in it, so a card carrying one has to lose it."""
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["data/a.parquet", "data/b.parquet"]

    resolved = seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs)

    assert resolved == "datasets/org/repo/data/a.parquet"


@pytest.mark.parametrize(
    ("split", "expected"),
    [("Train", "datasets/org/repo/b.parquet"), ("train", "datasets/org/repo/a.parquet")],
)
def test_seed_hf_path_reads_a_split_name_exactly(monkeypatch, tmp_path, split, expected):
    """load_dataset is handed the name the caller asked for, not a folded one."""
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["a.parquet", "b.parquet"]
    configs = [
        {
            "config_name": "default",
            "data_files": [
                {"split": "train", "path": "a.parquet"},
                {"split": "Train", "path": "b.parquet"},
            ],
        }
    ]

    assert seed_route._resolve_seed_hf_path("org/repo", files, split, None, configs) == expected


def test_seed_hf_path_reads_declared_files_under_data_dir_even_when_they_repeat_it(
    monkeypatch, tmp_path
):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["data/data/train.parquet", "data/train.parquet"]
    configs = [
        {
            "config_name": "default",
            "data_dir": "data",
            "data_files": [{"split": "train", "path": "data/train.parquet"}],
        }
    ]

    resolved = seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs)

    assert resolved == "datasets/org/repo/data/data/train.parquet"


def test_seed_hf_path_collapses_a_parent_component_in_a_card_path(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["data/a.parquet", "data/b.parquet"]
    configs = [
        {
            "config_name": "default",
            "data_dir": "staging/../data",
            "data_files": [
                {"split": "train", "path": "a.parquet"},
                {"split": "test", "path": "b.parquet"},
            ],
        }
    ]

    resolved = seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs)

    assert resolved == "datasets/org/repo/data/a.parquet"


def test_seed_hf_path_leaves_loader_ignored_metadata_out_of_a_card_glob(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["train.jsonl", "dataset_info.json"]
    configs = [{"config_name": "default", "data_files": [{"split": "train", "path": "*"}]}]

    resolved = seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs)

    matched = seed_route._files_under_patterns([resolved[len("datasets/org/repo/") :]], files)
    assert matched == ["train.jsonl"]


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [("[^a].parquet", ["a.parquet"]), ("[!a].parquet", ["b.parquet"])],
)
def test_seed_glob_reads_a_caret_in_a_class_literally(monkeypatch, tmp_path, pattern, expected):
    """fsspec negates on ! alone, so [^a] is the two characters ^ and a."""
    seed_route = _load_seed_route(monkeypatch, tmp_path)

    assert seed_route._files_under_patterns([pattern], ["a.parquet", "b.parquet"]) == expected


def test_seed_hf_path_keeps_a_dunder_file_a_card_declared(monkeypatch, tmp_path):
    """The loader skips a __folder, not a file whose name happens to start that way."""
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["data/__train.json", "data/train.json"]
    configs = [{"config_name": "default", "data_files": [{"split": "train", "path": "data/*"}]}]

    resolved = seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs)

    matched = seed_route._files_under_patterns([resolved[len("datasets/org/repo/") :]], files)
    assert sorted(matched) == ["data/__train.json", "data/train.json"]


def test_seed_config_named_Default_is_not_the_implicit_default(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)

    assert (
        seed_route._pick_config([{"config_name": "Default"}, {"config_name": "other"}], None)
        is None
    )


def test_seed_hf_path_ignores_an_extension_collision_out_of_reach(monkeypatch, tmp_path):
    """The glob is anchored at data/, so a name in other/ cannot spoil it."""
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["data/a.json", "data/b.jsonl"]
    configs = [{"config_name": "default", "data_files": [{"split": "train", "path": "data/*"}]}]

    resolved = seed_route._resolve_seed_hf_path(
        "org/repo", files, "train", None, configs, [*files, "other/c.json.gz"]
    )

    assert resolved == "datasets/org/repo/data/*.json*"


@pytest.mark.parametrize(
    ("subset", "expected"),
    [
        ("Foo", "datasets/org/repo/Foo/train*.parquet"),
        ("foo", "datasets/org/repo/foo/train*.parquet"),
    ],
)
def test_seed_hf_path_narrows_to_the_subset_folder_by_its_own_case(
    monkeypatch, tmp_path, subset, expected
):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["Foo/train.parquet", "foo/train.parquet"]

    assert seed_route._resolve_seed_hf_path("org/repo", files, "train", subset) == expected


def test_seed_hf_path_unions_an_exact_and_a_qualified_split_folder(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["train/0.parquet", "sets/train_a/1.parquet", "sets/test_b/2.parquet"]

    resolved = seed_route._resolve_seed_hf_path("org/repo", files, "train")

    matched = seed_route._files_under_patterns([resolved[len("datasets/org/repo/") :]], files)
    assert sorted(matched) == ["sets/train_a/1.parquet", "train/0.parquet"]


def test_seed_hf_path_keeps_folders_naming_the_split_in_the_middle(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = [
        "sets/a_train_x/0.parquet",
        "sets/b_train_y/0.parquet",
        "sets/c_test_z/0.parquet",
    ]

    resolved = seed_route._resolve_seed_hf_path("org/repo", files, "train")

    matched = seed_route._files_under_patterns([resolved[len("datasets/org/repo/") :]], files)
    assert sorted(matched) == ["sets/a_train_x/0.parquet", "sets/b_train_y/0.parquet"]


def test_seed_hf_path_keeps_qualified_split_folders_together(monkeypatch, tmp_path):
    """train_a and train_b are both train to the loader, so both must be read."""
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = [
        "sets/train_a/0.parquet",
        "sets/train_b/1.parquet",
        "sets/test_a/2.parquet",
    ]

    resolved = seed_route._resolve_seed_hf_path("org/repo", files, "train")

    matched = seed_route._files_under_patterns([resolved[len("datasets/org/repo/") :]], files)
    assert sorted(matched) == ["sets/train_a/0.parquet", "sets/train_b/1.parquet"]


def test_seed_hf_path_gathers_a_split_written_under_two_aliases(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["data/dev-0.parquet", "data/validation-0.parquet", "data/test-0.parquet"]

    resolved = seed_route._resolve_seed_hf_path("org/repo", files, "validation")

    matched = seed_route._files_under_patterns([resolved[len("datasets/org/repo/") :]], files)
    assert sorted(matched) == ["data/dev-0.parquet", "data/validation-0.parquet"]


def test_seed_hf_path_counts_json_and_jsonl_as_two_extensions(monkeypatch, tmp_path):
    """`load.infer_module_for_data_files_list` votes per extension, parquet on ties."""
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["data/a.json", "data/b.jsonl", "data/c.parquet"]
    configs = [{"config_name": "default", "data_files": [{"split": "train", "path": "data/*"}]}]

    resolved = seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs)

    assert resolved == "datasets/org/repo/data/*.parquet"


def test_seed_hf_path_keeps_shards_spread_over_sibling_folders(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["a/train-0.parquet", "b/train-1.parquet", "a/test-0.parquet"]

    resolved = seed_route._resolve_seed_hf_path("org/repo", files, "train")

    assert resolved == "datasets/org/repo/**/train-*.parquet"


def test_seed_hf_path_reads_sibling_folders_as_one_split_without_a_card(monkeypatch, tmp_path):
    """Without a card there are no configs, so every train file is the train split."""
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = [
        "main/train-0.parquet",
        "main/test-0.parquet",
        "socratic/train-0.parquet",
        "socratic/test-0.parquet",
    ]

    assert (
        seed_route._resolve_seed_hf_path("org/repo", files, "train")
        == "datasets/org/repo/**/train-*.parquet"
    )
    # Ask for one of them and only that one is read.
    assert (
        seed_route._resolve_seed_hf_path("org/repo", files, "train", "socratic")
        == "datasets/org/repo/socratic/train-*.parquet"
    )


def test_seed_preview_file_follows_the_config_data_dir(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["en/train.parquet", "fr/train.parquet"]
    configs = [{"config_name": "french", "data_dir": "fr"}]
    monkeypatch.setattr(seed_route, "_list_hf_repo_files", lambda **kwargs: files)
    monkeypatch.setattr(seed_route, "_list_hf_dataset_configs", lambda **kwargs: configs)
    monkeypatch.setattr(
        seed_route, "refuse_unauthorized_dataset_preview", lambda *args, **kwargs: None
    )
    seen: list[str | None] = []

    def fake_preview(*, load_dataset_fn, load_kwargs, preview_size):
        seen.append((load_kwargs.get("data_files") or [None])[0])
        return [{"text": "row"}]

    monkeypatch.setattr(seed_route, "_load_preview_rows", fake_preview)

    response = seed_route.inspect_seed_dataset(
        SimpleNamespace(
            dataset_name = "org/repo",
            split = "train",
            subset = "french",
            hf_token = None,
            preview_size = 1,
        ),
        allow_ambient_token = False,
    )

    assert seen[0] == "fr/train.parquet"
    assert response.resolved_path == "datasets/org/repo/fr/train*.parquet"


def test_seed_hf_path_treats_an_unnamed_config_as_the_default(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    configs = [
        {
            "data_files": [
                {"split": "train", "path": "a/part.parquet"},
                {"split": "test", "path": "b/part.parquet"},
            ]
        }
    ]
    files = ["a/part.parquet", "b/part.parquet"]
    assert (
        seed_route._resolve_seed_hf_path("org/repo", files, "test", "default", configs)
        == "datasets/org/repo/b/part.parquet"
    )


def test_seed_hf_path_ignores_a_subset_label_that_is_not_the_config(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    # x/main.parquet carries the label but nothing of the split, so it is a name
    # collision rather than the config.
    files = ["x/main.parquet", "y/train-0.parquet"]
    assert (
        seed_route._resolve_seed_hf_path("org/repo", files, "train", "main")
        == "datasets/org/repo/y/train-*.parquet"
    )


def test_seed_hf_path_keeps_the_split_when_widening_several_declared_globs(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["sets/a/train-0.parquet", "sets/b/train-0.parquet", "sets/a/test-0.parquet"]
    configs = [
        {
            "config_name": "default",
            "data_files": [
                # Declared twice for the same split, which must not lose the second.
                {"split": "train", "path": "sets/a/train-*"},
                {"split": "train", "path": "sets/b/train-*"},
            ],
        }
    ]
    assert seed_route._declared_split_patterns(configs, "train") == [
        "sets/a/train-*",
        "sets/b/train-*",
    ]
    # sets/**/*.parquet would take the test file back; the split-named form does not.
    assert (
        seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs)
        == "datasets/org/repo/sets/**/*train*.parquet"
    )


def test_seed_hf_path_keeps_a_split_whose_name_carries_a_separator(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = [
        "data/validation_matched-000.parquet",
        "data/validation_mismatched-000.parquet",
        "data/train-000.parquet",
    ]
    assert (
        seed_route._resolve_seed_hf_path("org/repo", files, "validation_matched")
        == "datasets/org/repo/data/validation_matched-*.parquet"
    )


def test_seed_hf_path_keeps_a_subset_whose_name_carries_a_separator(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["sample-10BT-train.parquet", "sample-100BT-train.parquet"]
    assert (
        seed_route._resolve_seed_hf_path("org/repo", files, "train", "sample-10BT")
        == "datasets/org/repo/sample-10BT-train*.parquet"
    )


def test_seed_hf_path_widens_through_split_named_folders(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["sets/train_a/0.parquet", "sets/train_b/0.parquet", "sets/test/0.parquet"]
    configs = [
        {
            "config_name": "default",
            "data_files": [
                {"split": "train", "path": ["sets/train_a/*", "sets/train_b/*"]},
                {"split": "test", "path": "sets/test/*"},
            ],
        }
    ]
    # The shard names say nothing, so the split has to come from the folders.
    assert (
        seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs)
        == "datasets/org/repo/sets/**/*train*/**/*.parquet"
    )


def test_seed_globstar_keeps_the_folder_boundary(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["data/train.csv", "data/z/train.csv", "data/nottrain.parquet"]
    assert seed_route._files_under_patterns(["data/**/train.*"], files) == [
        "data/train.csv",
        "data/z/train.csv",
    ]


def test_seed_declared_files_match_the_glob_not_its_prefix(monkeypatch, tmp_path):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["data/train-notes.json", "data/train-00000-of-00002.parquet"]
    assert seed_route._files_under_patterns(["data/train-*.parquet"], files) == [
        "data/train-00000-of-00002.parquet"
    ]
    assert seed_route._files_under_patterns(["data/*/x.parquet"], ["data/a/b/x.parquet"]) == []
    assert seed_route._files_under_patterns(["data/**/x.parquet"], ["data/a/b/x.parquet"]) == [
        "data/a/b/x.parquet"
    ]


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        # train-part.parquet wins on length, but questions_train.parquet is train
        # too, so the pattern has to reach both without taking test-part.
        (
            ["data/train-part.parquet", "data/questions_train.parquet", "data/test-part.parquet"],
            "datasets/org/repo/data/*train*.parquet",
        ),
        # A split named mid-name, the conventional sharded form.
        (
            ["data/questions_train_000.jsonl", "data/questions_test_000.jsonl"],
            "datasets/org/repo/data/*train*.jsonl",
        ),
        # A dotted split name: only the final extension comes off the stem.
        (
            ["data/questions.train.parquet", "data/questions.test.parquet"],
            "datasets/org/repo/data/*train*.parquet",
        ),
        # "training" is datasets' own alias for train, so both belong to it.
        (
            ["data/train-0.parquet", "data/training-0.parquet", "data/test-0.parquet"],
            "datasets/org/repo/data/*train*.parquet",
        ),
    ],
)
def test_seed_hf_path_reaches_every_file_of_the_split_and_no_other(
    monkeypatch, tmp_path, files, expected
):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    assert seed_route._resolve_seed_hf_path("org/repo", files, "train") == expected


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("data/train.*", "datasets/org/repo/data/train.parquet"),
        ("data/train-*.*", "datasets/org/repo/data/train-*.parquet"),
        ("data/train-*", "datasets/org/repo/data/train-*.parquet"),
        ("data/train-0.parquet", "datasets/org/repo/data/train-0.parquet"),
    ],
)
def test_seed_hf_path_gives_a_card_glob_a_readable_extension(
    monkeypatch, tmp_path, declared, expected
):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    files = ["data/train.parquet", "data/train-0.parquet", "data/test-0.parquet"]
    configs = [{"config_name": "default", "data_files": [{"split": "train", "path": declared}]}]
    assert seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs) == expected


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        (["part-0.parquet"], "datasets/org/repo/**/*.parquet"),
        (["data/part-0.parquet", "data/part-1.parquet"], "datasets/org/repo/data/**/*.parquet"),
        (
            ["data/test/0.parquet", "data/train/0.parquet"],
            "datasets/org/repo/data/train/**/*.parquet",
        ),
        # Split folders at the repo root: the shorter "test" path used to win on
        # length alone, so a train recipe was served the test split.
        (["test/0.parquet", "train/0.parquet"], "datasets/org/repo/train/**/*.parquet"),
    ],
)
def test_seed_hf_path_still_globs_the_directory_without_a_split_in_the_name(
    monkeypatch, tmp_path, files, expected
):
    seed_route = _load_seed_route(monkeypatch, tmp_path)
    assert seed_route._resolve_seed_hf_path("org/repo", files, "train") == expected


def test_validate_resolves_the_hf_seed_endpoint_like_jobs(monkeypatch):
    pytest.importorskip("fastapi")
    backend_root = Path(__file__).resolve().parent.parent
    monkeypatch.syspath_prepend(str(backend_root))
    monkeypatch.setenv("HF_ENDPOINT", "http://127.0.0.1:9700")

    from models.data_recipe import RecipePayload
    from routes.data_recipe import validate as validate_module

    seen = {}
    monkeypatch.setattr(validate_module, "validate_recipe", lambda recipe: seen.update(recipe))

    response = validate_module.validate(
        RecipePayload(
            recipe = {
                "seed_config": {
                    "source": {
                        "seed_type": "hf",
                        "path": "datasets/a/b/**/*.parquet",
                        "endpoint": None,
                    }
                },
                "columns": [{"column_type": "expression", "name": "x", "expr": "{{ q }}"}],
            }
        )
    )

    assert response.valid is True
    assert seen["seed_config"]["source"]["endpoint"] == "http://127.0.0.1:9700"

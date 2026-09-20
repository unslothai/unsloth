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
        (_GSM8K_FILES, "train", None, "datasets/org/repo/main/train-*.parquet"),
        (_GSM8K_FILES, "test", "main", "datasets/org/repo/main/test-*.parquet"),
        (_GSM8K_FILES, "train", "socratic", "datasets/org/repo/socratic/train-*.parquet"),
        (_GSM8K_FILES, "train", "default", "datasets/org/repo/main/train-*.parquet"),
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
    # than dropping one of them.
    assert (
        seed_route._resolve_seed_hf_path("org/repo", files, "train", None, configs)
        == "datasets/org/repo/sets/**/*.parquet"
    )


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
        # "training" is not the train split, so the narrow prefix still wins.
        (
            ["data/train-0.parquet", "data/training-0.parquet", "data/test-0.parquet"],
            "datasets/org/repo/data/train-*.parquet",
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

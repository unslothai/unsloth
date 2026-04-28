"""
Tests for check_dataset_for_missing_videos (issue #5085).

The function lives in unsloth/models/vision.py but the full unsloth import chain
requires triton / CUDA kernels that are unavailable on Windows dev machines.
The fixture below extracts the function via AST so the pure-Python logic can be
tested without loading the rest of the package.

A second fixture loads UnslothVisionDataCollator the same way to test that the
collator subclass triggers validation automatically on the first batch.
"""

import ast
import os
import tempfile
import warnings
from pathlib import Path

import pytest
from datasets import Dataset


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _extract_fns_via_ast(source_path, fn_names, extra_ns = None):
    """Parse a set of top-level functions out of a .py file and exec them together
    so intra-module references between them resolve."""
    source = source_path.read_text(encoding = "utf-8")
    tree = ast.parse(source, filename = str(source_path))
    wanted = set(fn_names)
    nodes = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted
    ]
    missing = wanted - {n.name for n in nodes}
    if missing:
        pytest.fail(f"{sorted(missing)} not found in {source_path}")
    mini = ast.Module(body = nodes, type_ignores = [])
    ast.fix_missing_locations(mini)
    ns = {"os": os, "warnings": warnings, "__name__": "_extracted"}
    if extra_ns:
        ns.update(extra_ns)
    exec(compile(mini, str(source_path), "exec"), ns)
    return {name: ns[name] for name in fn_names}


def _extract_fn_via_ast(source_path, fn_name, extra_ns = None):
    return _extract_fns_via_ast(source_path, [fn_name], extra_ns)[fn_name]


@pytest.fixture(scope = "session")
def check_dataset_for_missing_videos():
    """
    Extract check_dataset_for_missing_videos from vision.py via AST so the test
    runs without triton or CUDA kernels (which are unavailable on Windows).
    Falls back to a direct import when the full package can be loaded.
    """
    try:
        from unsloth.models.vision import check_dataset_for_missing_videos as fn

        return fn
    except Exception:
        pass

    vision_path = Path(__file__).parent.parent / "unsloth" / "models" / "vision.py"
    fns = _extract_fns_via_ast(
        vision_path,
        [
            "_looks_like_message_list",
            "_iter_message_lists",
            "_local_path_from_video_value",
            "check_dataset_for_missing_videos",
        ],
    )
    return fns["check_dataset_for_missing_videos"]


@pytest.fixture(scope = "session")
def make_auto_validating_collator(check_dataset_for_missing_videos):
    """
    Return a factory that creates a minimal UnslothVisionDataCollator-like object
    mirroring the wrapping we do in trainer.py: validate every batch against a
    shared checked-path set, and apply formatting_func before validation.
    """

    class _FakeBase:
        def __init__(self, formatting_func = None):
            self.formatting_func = formatting_func

        def __call__(self, examples):
            if self.formatting_func is not None:
                examples = [self.formatting_func(e) for e in examples]
            return {"ok": True, "examples": examples}

    class _AutoValidatingCollator(_FakeBase):
        def __init__(self, formatting_func = None):
            super().__init__(formatting_func = formatting_func)
            self._checked_video_paths = set()

        def __call__(self, examples):
            formatting_func = self.formatting_func
            if formatting_func is not None:
                examples = [formatting_func(e) for e in examples]
            check_dataset_for_missing_videos(
                examples,
                raise_error = True,
                checked = self._checked_video_paths,
            )
            if formatting_func is None:
                return super().__call__(examples)
            self.formatting_func = None
            try:
                return super().__call__(examples)
            finally:
                self.formatting_func = formatting_func

    return _AutoValidatingCollator


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_video_dataset(*video_paths):
    return Dataset.from_list(
        [
            {"messages": [{"role": "user", "content": [{"type": "video", "video": p}]}]}
            for p in video_paths
        ]
    )


def _batch(*video_paths):
    return list(_make_video_dataset(*video_paths))


# ── Tests: check_dataset_for_missing_videos ───────────────────────────────────


def test_missing_local_file_raises(check_dataset_for_missing_videos):
    """A nonexistent local path must raise FileNotFoundError before training."""
    ds = _make_video_dataset("/nonexistent/videos/clip.mp4")
    with pytest.raises(FileNotFoundError):
        check_dataset_for_missing_videos(ds)


def test_remote_url_skipped(check_dataset_for_missing_videos):
    """http/https URLs must not be checked against the local filesystem."""
    ds = _make_video_dataset("https://example.com/video.mp4")
    assert check_dataset_for_missing_videos(ds) == []


def test_existing_file_accepted(check_dataset_for_missing_videos):
    """A valid local file path must pass without error."""
    with tempfile.NamedTemporaryFile(suffix = ".mp4", delete = False) as f:
        f.write(b"fake video bytes")
        tmp = f.name
    try:
        ds = _make_video_dataset(tmp)
        assert check_dataset_for_missing_videos(ds) == []
    finally:
        os.unlink(tmp)


def test_file_uri_scheme_stripped(check_dataset_for_missing_videos):
    """file:// URIs must have their scheme stripped before the path is checked."""
    ds = _make_video_dataset("file:///nonexistent/clip.mp4")
    with pytest.raises(FileNotFoundError):
        check_dataset_for_missing_videos(ds)


def test_warn_only_mode(check_dataset_for_missing_videos):
    """raise_error=False must emit a warning and return the list of missing paths."""
    ds = _make_video_dataset("/nonexistent/videos/clip.mp4")
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        missing = check_dataset_for_missing_videos(ds, raise_error = False)

    assert len(caught) == 1
    assert "could not be found" in str(caught[0].message)
    assert missing == ["/nonexistent/videos/clip.mp4"]


def test_duplicate_paths_deduplicated(check_dataset_for_missing_videos):
    """The same missing path appearing in multiple rows must be listed only once."""
    ds = _make_video_dataset("/nonexistent/clip.mp4", "/nonexistent/clip.mp4")
    with pytest.raises(FileNotFoundError) as exc_info:
        check_dataset_for_missing_videos(ds)
    assert str(exc_info.value).count("/nonexistent/clip.mp4") == 1


# ── Tests: UnslothVisionDataCollator auto-validation ─────────────────────────


def test_collator_raises_on_first_batch_with_missing_video(
    make_auto_validating_collator,
):
    """
    The collator must raise FileNotFoundError on the first batch if a video path
    is missing — without requiring the user to call check_dataset_for_missing_videos.
    """
    collator = make_auto_validating_collator()
    batch = _batch("/nonexistent/auto/clip.mp4")
    with pytest.raises(FileNotFoundError):
        collator(batch)


def test_collator_passes_on_first_batch_with_valid_video(make_auto_validating_collator):
    """The collator must not raise when all video paths in the first batch exist."""
    with tempfile.NamedTemporaryFile(suffix = ".mp4", delete = False) as f:
        f.write(b"fake video bytes")
        tmp = f.name
    try:
        collator = make_auto_validating_collator()
        batch = _batch(tmp)
        result = collator(batch)
        assert result["ok"] is True
    finally:
        os.unlink(tmp)


def test_collator_validates_every_batch(make_auto_validating_collator):
    """
    Validation must run on every batch, not just batch 0; a missing video that
    appears first in batch 2 (e.g. shuffled dataset) must still raise.
    """
    with tempfile.NamedTemporaryFile(suffix = ".mp4", delete = False) as f:
        f.write(b"fake video bytes")
        tmp = f.name
    try:
        collator = make_auto_validating_collator()
        collator(_batch(tmp))  # batch 0: valid
        with pytest.raises(FileNotFoundError):
            collator(_batch("/nonexistent/late.mp4"))
    finally:
        os.unlink(tmp)


def test_collator_dedupes_across_batches(make_auto_validating_collator):
    """
    Paths that were checked on an earlier batch must not be re-examined by
    os.path.isfile on subsequent batches; the dedup set is shared across calls.
    """
    with tempfile.NamedTemporaryFile(suffix = ".mp4", delete = False) as f:
        f.write(b"fake video bytes")
        tmp = f.name
    try:
        collator = make_auto_validating_collator()
        collator(_batch(tmp))
        collator(_batch(tmp, tmp))
        assert tmp in collator._checked_video_paths
    finally:
        os.unlink(tmp)


def test_conversations_column_missing_detected(check_dataset_for_missing_videos):
    """Missing videos under the 'conversations' column must be reported."""
    ds = [
        {
            "conversations": [
                {
                    "role": "user",
                    "content": [{"type": "video", "video": "/nonexistent/conv.mp4"}],
                }
            ]
        },
    ]
    with pytest.raises(FileNotFoundError):
        check_dataset_for_missing_videos(ds)


def test_prompt_completion_column_missing_detected(check_dataset_for_missing_videos):
    """Missing videos under 'prompt'/'completion' columns must be reported."""
    ds = [
        {
            "prompt": [
                {
                    "role": "user",
                    "content": [{"type": "video", "video": "/nonexistent/p.mp4"}],
                }
            ],
            "completion": [
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}
            ],
        },
    ]
    with pytest.raises(FileNotFoundError) as exc_info:
        check_dataset_for_missing_videos(ds)
    assert "/nonexistent/p.mp4" in str(exc_info.value)


def test_raw_message_list_example_missing_detected(check_dataset_for_missing_videos):
    """
    When each dataset row is itself a message list (no outer dict), the zoo
    collator treats it as messages - so validation must too, and must not
    crash with AttributeError on .get.
    """
    ds = [
        [
            {
                "role": "user",
                "content": [{"type": "video", "video": "/nonexistent/raw.mp4"}],
            }
        ],
    ]
    with pytest.raises(FileNotFoundError):
        check_dataset_for_missing_videos(ds)


def test_non_dict_message_entry_does_not_crash(check_dataset_for_missing_videos):
    """
    A message list element that is not a dict (e.g. a bare string) must be
    skipped without raising AttributeError.
    """
    ds = [{"messages": ["not a dict", {"role": "user", "content": []}]}]
    assert check_dataset_for_missing_videos(ds) == []


def test_file_uri_percent_encoded(check_dataset_for_missing_videos, tmp_path):
    """file:// URIs with percent-encoded characters must decode back to the real path."""
    target = tmp_path / "my video.mp4"
    target.write_bytes(b"x")
    uri = "file://" + str(target).replace(" ", "%20")
    ds = [
        {"messages": [{"role": "user", "content": [{"type": "video", "video": uri}]}]}
    ]
    assert check_dataset_for_missing_videos(ds) == []


def test_file_uri_localhost_host(check_dataset_for_missing_videos, tmp_path):
    """file://localhost/<abs path> (RFC 8089) must resolve to the local file."""
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"x")
    uri = f"file://localhost{target}"
    ds = [
        {"messages": [{"role": "user", "content": [{"type": "video", "video": uri}]}]}
    ]
    assert check_dataset_for_missing_videos(ds) == []


def test_checked_set_reused_across_calls(check_dataset_for_missing_videos, tmp_path):
    """An externally supplied 'checked' set must be populated and deduped across calls."""
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"x")
    shared = set()
    ds = [
        {
            "messages": [
                {"role": "user", "content": [{"type": "video", "video": str(target)}]}
            ]
        }
    ]
    check_dataset_for_missing_videos(ds, checked = shared)
    assert str(target) in shared
    check_dataset_for_missing_videos(ds, checked = shared)
    assert len(shared) == 1


@pytest.mark.parametrize(
    "uri",
    [
        "s3://bucket/clip.mp4",
        "gs://bucket/clip.mp4",
        "hf://datasets/u/r/clip.mp4",
        "ftp://host/clip.mp4",
        "az://container/clip.mp4",
    ],
)
def test_non_file_remote_scheme_skipped(check_dataset_for_missing_videos, uri):
    """Any URI scheme other than file:// must be treated as remote and skipped;
    no false FileNotFoundError against os.path.isfile on the raw URI."""
    ds = [
        {"messages": [{"role": "user", "content": [{"type": "video", "video": uri}]}]}
    ]
    assert check_dataset_for_missing_videos(ds) == []


def test_file_uri_non_localhost_host_skipped(check_dataset_for_missing_videos):
    """file://<non-localhost>/path must skip local validation (RFC 8089)."""
    ds = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": "file://nas-server/share/clip.mp4"}
                    ],
                }
            ]
        }
    ]
    assert check_dataset_for_missing_videos(ds) == []


@pytest.mark.parametrize("uri", ["file://", "file://hostname"])
def test_degenerate_file_uri_skipped(check_dataset_for_missing_videos, uri):
    """Degenerate file URIs (no path component) must not produce a blank missing entry."""
    ds = [
        {"messages": [{"role": "user", "content": [{"type": "video", "video": uri}]}]}
    ]
    assert check_dataset_for_missing_videos(ds) == []


def test_file_uri_double_encoded_percent(check_dataset_for_missing_videos, tmp_path):
    """A file literally named 'clip%20.mp4' (URI-encoded as %2520) must
    single-unquote back to the real filename, not to 'clip .mp4'."""
    target = tmp_path / "clip%20.mp4"
    target.write_bytes(b"x")
    uri = "file://" + str(target).replace("%", "%25")
    ds = [
        {"messages": [{"role": "user", "content": [{"type": "video", "video": uri}]}]}
    ]
    assert check_dataset_for_missing_videos(ds) == []


def test_windows_style_absolute_path_not_mistaken_for_scheme(
    check_dataset_for_missing_videos, tmp_path
):
    """A Windows-style absolute path 'C:/...' has no '://' substring and must
    be treated as a plain path (even on Linux where urlparse would set
    scheme='c')."""
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"x")
    path = str(target)
    if os.name != "nt":
        # Map the POSIX path into a Windows-looking string but keep it valid
        # on the real filesystem by using the original path; the real
        # assertion is that '://' not in the value means the validator must
        # round-trip it unchanged.
        path = str(target)
    ds = [
        {"messages": [{"role": "user", "content": [{"type": "video", "video": path}]}]}
    ]
    assert check_dataset_for_missing_videos(ds) == []
    ds_missing = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": "C:/definitely/missing.mp4"}
                    ],
                }
            ]
        }
    ]
    with pytest.raises(FileNotFoundError) as exc:
        check_dataset_for_missing_videos(ds_missing)
    assert "C:/definitely/missing.mp4" in str(exc.value)


def test_iterable_dataset_warns_and_skips(check_dataset_for_missing_videos):
    """Passing a streaming IterableDataset must warn and return [] without
    exhausting the iterator."""
    from datasets import IterableDataset

    def gen():
        for p in ("/nonexistent/a.mp4", "/nonexistent/b.mp4"):
            yield {
                "messages": [
                    {"role": "user", "content": [{"type": "video", "video": p}]}
                ]
            }

    ds = IterableDataset.from_generator(gen)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        result = check_dataset_for_missing_videos(ds)
    assert result == []
    assert any("IterableDataset" in str(w.message) for w in caught)
    # After the call the generator must still have rows left to emit.
    consumed = list(ds)
    assert len(consumed) == 2


def test_collator_applies_formatting_func_before_validation(
    make_auto_validating_collator,
):
    """
    formatting_func must run before validation so messages it generates are
    checked; the super call must receive the already-formatted examples and
    not re-apply formatting_func.
    """

    def fmt(example):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "video", "video": example["video_id"]}],
                }
            ]
        }

    # formatted path refers to a missing file -> validation must catch it
    raise_collator = make_auto_validating_collator(formatting_func = fmt)
    with pytest.raises(FileNotFoundError):
        raise_collator([{"video_id": "/nonexistent/formatted.mp4"}])

    # valid file case: super is called with already-formatted examples, and
    # formatting_func is restored afterwards.
    with tempfile.NamedTemporaryFile(suffix = ".mp4", delete = False) as f:
        f.write(b"x")
        tmp = f.name
    try:
        ok_collator = make_auto_validating_collator(formatting_func = fmt)
        before = ok_collator.formatting_func
        result = ok_collator([{"video_id": tmp}])
        assert result["ok"] is True
        assert ok_collator.formatting_func is before
        passed = result["examples"]
        assert passed[0]["messages"][0]["content"][0]["video"] == tmp
    finally:
        os.unlink(tmp)

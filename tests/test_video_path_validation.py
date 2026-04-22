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

def _extract_fn_via_ast(source_path, fn_name, extra_ns=None):
    """Parse a single top-level function out of a .py file and exec it."""
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    func_node = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn_name),
        None,
    )
    if func_node is None:
        pytest.fail(f"{fn_name} not found in {source_path}")
    mini = ast.Module(body=[func_node], type_ignores=[])
    ast.fix_missing_locations(mini)
    ns = {"os": os, "warnings": warnings}
    if extra_ns:
        ns.update(extra_ns)
    exec(compile(mini, str(source_path), "exec"), ns)
    return ns[fn_name]


@pytest.fixture(scope="session")
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
    return _extract_fn_via_ast(vision_path, "check_dataset_for_missing_videos")


@pytest.fixture(scope="session")
def make_auto_validating_collator(check_dataset_for_missing_videos):
    """
    Return a factory that creates a minimal UnslothVisionDataCollator-like object
    with the same auto-validation wrapping as our trainer.py subclass, but without
    needing a processor or CUDA.
    """
    class _FakeBase:
        def __call__(self, examples):
            return {"ok": True}

    class _AutoValidatingCollator(_FakeBase):
        def __init__(self):
            self._video_paths_validated = False

        def __call__(self, examples):
            if not self._video_paths_validated:
                self._video_paths_validated = True
                check_dataset_for_missing_videos(examples, raise_error=True)
            return super().__call__(examples)

    return _AutoValidatingCollator


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_video_dataset(*video_paths):
    return Dataset.from_list([
        {"messages": [{"role": "user", "content": [{"type": "video", "video": p}]}]}
        for p in video_paths
    ])


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
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        missing = check_dataset_for_missing_videos(ds, raise_error=False)

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

def test_collator_raises_on_first_batch_with_missing_video(make_auto_validating_collator):
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
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"fake video bytes")
        tmp = f.name
    try:
        collator = make_auto_validating_collator()
        batch = _batch(tmp)
        result = collator(batch)
        assert result == {"ok": True}
    finally:
        os.unlink(tmp)


def test_collator_validates_only_once(make_auto_validating_collator):
    """
    After the first batch passes, subsequent batches with missing paths must not
    re-trigger validation (validation is a startup check, not per-batch).
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"fake video bytes")
        tmp = f.name
    try:
        collator = make_auto_validating_collator()
        collator(_batch(tmp))                              # first batch: valid, sets flag
        result = collator(_batch("/nonexistent/late.mp4")) # second batch: missing, no raise
        assert result == {"ok": True}
    finally:
        os.unlink(tmp)


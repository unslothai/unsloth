"""
Tests for check_dataset_for_missing_videos (issue #5085).

The function lives in unsloth/models/vision.py but the full unsloth import chain
requires triton / CUDA kernels that are unavailable on Windows dev machines.
The fixture below extracts the function via AST so the pure-Python logic can be
tested without loading the rest of the package.
"""

import ast
import os
import tempfile
import warnings
from pathlib import Path

import pytest
from datasets import Dataset


# ── Fixture: load check_dataset_for_missing_videos without the full import ────

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
    source = vision_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(vision_path))

    func_node = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check_dataset_for_missing_videos"),
        None,
    )
    if func_node is None:
        pytest.fail("check_dataset_for_missing_videos not found in unsloth/models/vision.py")

    mini = ast.Module(body=[func_node], type_ignores=[])
    ast.fix_missing_locations(mini)
    ns = {"os": os, "warnings": warnings}
    exec(compile(mini, str(vision_path), "exec"), ns)
    return ns["check_dataset_for_missing_videos"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_video_dataset(*video_paths):
    return Dataset.from_list([
        {"messages": [{"role": "user", "content": [{"type": "video", "video": p}]}]}
        for p in video_paths
    ])


# ── Tests ─────────────────────────────────────────────────────────────────────

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

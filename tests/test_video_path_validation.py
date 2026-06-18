"""Tests for check_dataset_for_missing_videos (issue #5085).

Fixtures AST-extract the function from vision.py so logic tests run without
the full unsloth import chain (triton/CUDA kernels).
"""

import ast
import os
import tempfile
import warnings
from pathlib import Path

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _extract_fns_via_ast(
    source_path,
    fn_names,
    extra_ns = None,
):
    """Exec the named top-level functions from a .py file so their mutual references resolve."""
    source = source_path.read_text(encoding = "utf-8")
    tree = ast.parse(source, filename = str(source_path))
    wanted = set(fn_names)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
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


def _extract_fn_via_ast(
    source_path,
    fn_name,
    extra_ns = None,
):
    return _extract_fns_via_ast(source_path, [fn_name], extra_ns)[fn_name]


@pytest.fixture(scope = "session")
def check_dataset_for_missing_videos():
    """Direct import when possible, else AST extraction from vision.py."""
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
    """Factory for a minimal collator mirroring the trainer.py wrapper."""

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
    return [
        {"messages": [{"role": "user", "content": [{"type": "video", "video": p}]}]}
        for p in video_paths
    ]


def _batch(*video_paths):
    return _make_video_dataset(*video_paths)


# ── Tests: check_dataset_for_missing_videos ───────────────────────────────────


def test_missing_local_file_raises(check_dataset_for_missing_videos):
    """Missing local path raises FileNotFoundError."""
    ds = _make_video_dataset("/nonexistent/videos/clip.mp4")
    with pytest.raises(FileNotFoundError):
        check_dataset_for_missing_videos(ds)


def test_remote_url_skipped(check_dataset_for_missing_videos):
    """http/https URLs are not checked locally."""
    ds = _make_video_dataset("https://example.com/video.mp4")
    assert check_dataset_for_missing_videos(ds) == []


def test_existing_file_accepted(check_dataset_for_missing_videos):
    """Existing local file passes without error."""
    with tempfile.NamedTemporaryFile(suffix = ".mp4", delete = False) as f:
        f.write(b"fake video bytes")
        tmp = f.name
    try:
        ds = _make_video_dataset(tmp)
        assert check_dataset_for_missing_videos(ds) == []
    finally:
        os.unlink(tmp)


def test_file_uri_scheme_stripped(check_dataset_for_missing_videos):
    """file:// scheme is stripped before the path check."""
    ds = _make_video_dataset("file:///nonexistent/clip.mp4")
    with pytest.raises(FileNotFoundError):
        check_dataset_for_missing_videos(ds)


def test_warn_only_mode(check_dataset_for_missing_videos):
    """raise_error=False warns and returns the missing paths."""
    ds = _make_video_dataset("/nonexistent/videos/clip.mp4")
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        missing = check_dataset_for_missing_videos(ds, raise_error = False)

    assert len(caught) == 1
    assert "could not be found" in str(caught[0].message)
    assert missing == ["/nonexistent/videos/clip.mp4"]


def test_duplicate_paths_deduplicated(check_dataset_for_missing_videos):
    """Repeated missing path is listed once."""
    ds = _make_video_dataset("/nonexistent/clip.mp4", "/nonexistent/clip.mp4")
    with pytest.raises(FileNotFoundError) as exc_info:
        check_dataset_for_missing_videos(ds)
    assert str(exc_info.value).count("/nonexistent/clip.mp4") == 1


# ── Tests: UnslothVisionDataCollator auto-validation ─────────────────────────


def test_collator_raises_on_first_batch_with_missing_video(make_auto_validating_collator):
    """Collator raises on a missing path with no user action needed."""
    collator = make_auto_validating_collator()
    batch = _batch("/nonexistent/auto/clip.mp4")
    with pytest.raises(FileNotFoundError):
        collator(batch)


def test_collator_passes_on_first_batch_with_valid_video(make_auto_validating_collator):
    """Collator passes a valid batch through."""
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
    """A missing video first appearing after batch 0 must still raise."""
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
    """The checked-path set is shared across batches."""
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
    """'conversations' column is scanned."""
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
    """'prompt'/'completion' columns are scanned."""
    ds = [
        {
            "prompt": [
                {
                    "role": "user",
                    "content": [{"type": "video", "video": "/nonexistent/p.mp4"}],
                }
            ],
            "completion": [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}],
        },
    ]
    with pytest.raises(FileNotFoundError) as exc_info:
        check_dataset_for_missing_videos(ds)
    assert "/nonexistent/p.mp4" in str(exc_info.value)


def test_raw_message_list_example_missing_detected(check_dataset_for_missing_videos):
    """Rows that are themselves message lists (no outer dict) are scanned."""
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
    """Non-dict message entries are skipped."""
    ds = [{"messages": ["not a dict", {"role": "user", "content": []}]}]
    assert check_dataset_for_missing_videos(ds) == []


def test_file_uri_percent_encoded(check_dataset_for_missing_videos, tmp_path):
    """Percent-encoded file:// URIs decode to the real path."""
    target = tmp_path / "my video.mp4"
    target.write_bytes(b"x")
    uri = "file://" + str(target).replace(" ", "%20")
    ds = [{"messages": [{"role": "user", "content": [{"type": "video", "video": uri}]}]}]
    assert check_dataset_for_missing_videos(ds) == []


def test_file_uri_localhost_host(check_dataset_for_missing_videos, tmp_path):
    """file://localhost/<abs path> is the local machine (RFC 8089)."""
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"x")
    uri = f"file://localhost{target}"
    ds = [{"messages": [{"role": "user", "content": [{"type": "video", "video": uri}]}]}]
    assert check_dataset_for_missing_videos(ds) == []


def test_checked_set_reused_across_calls(check_dataset_for_missing_videos, tmp_path):
    """A supplied checked set is populated and deduped across calls."""
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"x")
    shared = set()
    ds = [{"messages": [{"role": "user", "content": [{"type": "video", "video": str(target)}]}]}]
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
    """Non-file URI schemes are treated as remote and skipped."""
    ds = [{"messages": [{"role": "user", "content": [{"type": "video", "video": uri}]}]}]
    assert check_dataset_for_missing_videos(ds) == []


def test_file_uri_non_localhost_host_skipped(check_dataset_for_missing_videos):
    """file://<non-localhost>/path is remote (RFC 8089): skip local checks."""
    ds = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "video", "video": "file://nas-server/share/clip.mp4"}],
                }
            ]
        }
    ]
    assert check_dataset_for_missing_videos(ds) == []


@pytest.mark.parametrize("uri", ["file://", "file://hostname"])
def test_degenerate_file_uri_skipped(check_dataset_for_missing_videos, uri):
    """No path component must not produce a blank missing entry."""
    ds = [{"messages": [{"role": "user", "content": [{"type": "video", "video": uri}]}]}]
    assert check_dataset_for_missing_videos(ds) == []


def test_file_uri_double_encoded_percent(check_dataset_for_missing_videos, tmp_path):
    """%2520 must single-unquote to 'clip%20.mp4', not 'clip .mp4'."""
    target = tmp_path / "clip%20.mp4"
    target.write_bytes(b"x")
    uri = "file://" + str(target).replace("%", "%25")
    ds = [{"messages": [{"role": "user", "content": [{"type": "video", "video": uri}]}]}]
    assert check_dataset_for_missing_videos(ds) == []


def test_windows_style_absolute_path_not_mistaken_for_scheme(
    check_dataset_for_missing_videos, tmp_path
):
    """'C:/...' has no '://' so it is a plain path, even where urlparse
    would yield scheme='c'."""
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"x")
    path = str(target)
    if os.name != "nt":
        # '://'-free values must round-trip unchanged; keep the real path
        path = str(target)
    ds = [{"messages": [{"role": "user", "content": [{"type": "video", "video": path}]}]}]
    assert check_dataset_for_missing_videos(ds) == []
    ds_missing = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "video", "video": "C:/definitely/missing.mp4"}],
                }
            ]
        }
    ]
    with pytest.raises(FileNotFoundError) as exc:
        check_dataset_for_missing_videos(ds_missing)
    assert "C:/definitely/missing.mp4" in str(exc.value)


def test_iterable_dataset_warns_and_skips(check_dataset_for_missing_videos):
    """Streaming IterableDataset: warn, return [], do not exhaust it."""
    datasets_mod = pytest.importorskip("datasets", reason = "real datasets package required")
    if not hasattr(datasets_mod, "IterableDataset"):
        pytest.skip("datasets.IterableDataset not available in this environment")
    IterableDataset = datasets_mod.IterableDataset

    def gen():
        for p in ("/nonexistent/a.mp4", "/nonexistent/b.mp4"):
            yield {"messages": [{"role": "user", "content": [{"type": "video", "video": p}]}]}

    ds = IterableDataset.from_generator(gen)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        result = check_dataset_for_missing_videos(ds)
    assert result == []
    assert any("IterableDataset" in str(w.message) for w in caught)
    # generator must not have been exhausted
    consumed = list(ds)
    assert len(consumed) == 2


def test_collator_applies_formatting_func_before_validation(make_auto_validating_collator):
    """formatting_func runs before validation; super gets formatted examples
    and must not re-apply it."""

    def fmt(example):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "video", "video": example["video_id"]}],
                }
            ]
        }

    raise_collator = make_auto_validating_collator(formatting_func = fmt)
    with pytest.raises(FileNotFoundError):
        raise_collator([{"video_id": "/nonexistent/formatted.mp4"}])

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


def test_data_uri_skipped(check_dataset_for_missing_videos):
    """Inline data: URIs are not flagged missing."""
    ds = _make_video_dataset("data:video/mp4;base64,AAAABBBBCCCC")
    assert check_dataset_for_missing_videos(ds) == []


def test_tuple_content_entries_checked(check_dataset_for_missing_videos):
    """Tuple message content is validated like a list."""
    ds = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": ({"type": "video", "video": "/nonexistent/tuple.mp4"},),
                }
            ]
        }
    ]
    with pytest.raises(FileNotFoundError):
        check_dataset_for_missing_videos(ds)


def test_duplicate_missing_deduped_in_warn_mode(check_dataset_for_missing_videos):
    """Warn mode returns each missing path once."""
    ds = _make_video_dataset("/nonexistent/dup.mp4", "/nonexistent/dup.mp4")
    with warnings.catch_warnings(record = True):
        warnings.simplefilter("always")
        missing = check_dataset_for_missing_videos(ds, raise_error = False)
    assert missing == ["/nonexistent/dup.mp4"]


# ── Tests: real unsloth_zoo collator integration ─────────────────────────────
# Exercise the real trainer.py subclass against the real zoo base (the fakes
# above don't cover super()/formatting_func); skip when unsloth can't import.


@pytest.fixture(scope = "session")
def real_collator_classes():
    try:
        from unsloth.trainer import UnslothVisionDataCollator
        from unsloth_zoo.vision_utils import (
            UnslothVisionDataCollator as ZooBase,
        )
    except Exception as exc:  # noqa: BLE001 - skip on any import failure
        pytest.skip(f"full unsloth import unavailable: {exc!r}")
    return UnslothVisionDataCollator, ZooBase


def _make_real_collator(real_collator_classes, formatting_func = None):
    """Build the real subclass without its heavy __init__ (needs a processor)."""
    subclass, _ = real_collator_classes
    collator = subclass.__new__(subclass)
    collator.formatting_func = formatting_func
    collator._checked_video_paths = set()
    return collator


def test_real_collator_blocks_super_on_missing_video(real_collator_classes, monkeypatch):
    """Missing path raises before the base __call__ runs."""
    _, zoo_base = real_collator_classes
    calls = []
    monkeypatch.setattr(zoo_base, "__call__", lambda self, examples: calls.append(examples))
    collator = _make_real_collator(real_collator_classes)
    with pytest.raises(FileNotFoundError):
        collator(_batch("/nonexistent/real.mp4"))
    assert calls == []  # base collator was never reached


def test_real_collator_calls_super_with_formatting_disabled(real_collator_classes, monkeypatch):
    """Base must see formatting_func=None and already-formatted examples;
    the original formatting_func is restored afterwards."""
    seen = {}

    def spy(self, examples):
        seen["formatting_func"] = self.formatting_func
        seen["examples"] = examples
        return {"ok": True}

    _, zoo_base = real_collator_classes
    monkeypatch.setattr(zoo_base, "__call__", spy)

    def fmt(example):
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "video", "video": example["video_id"]}],
                }
            ]
        }

    with tempfile.NamedTemporaryFile(suffix = ".mp4", delete = False) as f:
        f.write(b"x")
        tmp = f.name
    try:
        collator = _make_real_collator(real_collator_classes, formatting_func = fmt)
        result = collator([{"video_id": tmp}])
        assert result == {"ok": True}
        assert seen["formatting_func"] is None
        assert seen["examples"][0]["messages"][0]["content"][0]["video"] == tmp
        assert collator.formatting_func is fmt
        assert tmp in collator._checked_video_paths
    finally:
        os.unlink(tmp)


def test_real_collator_restores_formatting_func_when_super_raises(
    real_collator_classes, monkeypatch
):
    """formatting_func is restored even when the base raises."""

    def boom(self, examples):
        raise RuntimeError("base collator failed")

    _, zoo_base = real_collator_classes
    monkeypatch.setattr(zoo_base, "__call__", boom)

    def fmt(example):
        return {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}

    collator = _make_real_collator(real_collator_classes, formatting_func = fmt)
    with pytest.raises(RuntimeError):
        collator([{"anything": 1}])
    assert collator.formatting_func is fmt

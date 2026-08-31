# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""_preflight_first_batch rejects an empty/non-integer first batch (the base-model
empty-chat-template crash) before train(). The real methods are bound onto a light
fake self so the production logic runs against controlled batches."""

import contextlib
import importlib
import json
import os
import queue
import subprocess
import sys
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")


_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    """Register a stub module for a dep the CPU backend CI job does not install.

    The pytest job has studio.txt + torch + transformers but not unsloth/trl,
    which core.training.trainer imports at module scope. Stub the absent ones
    (real installs are left alone) so importing it for the two pure helper
    methods never breaks test collection. __spec__ = None keeps the trainer's
    own _ensure_real_packages namespace-shadow guard a no-op on the stub.
    """
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:
        pass
    _STUBBED.append(name)
    mod = types.ModuleType(name)
    mod.__spec__ = None
    for attr in attrs:
        setattr(mod, attr, MagicMock())
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, mod)


_STUB_SPECS = (
    ("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported")),
    ("unsloth.chat_templates", ("get_chat_template",)),
    ("trl", ("SFTTrainer", "SFTConfig")),
)


@contextlib.contextmanager
def _stubbed():
    """Hold the stubs for the duration of an import of the trainer, then drop them again.

    Leaving them in sys.modules outlives this module and the rest of the suite then runs against
    them: utils.hardware.hardware._shared_policy branches on `"unsloth" in sys.modules` and then
    reaches for unsloth.dataset_num_proc, which a spec-less non-package stub cannot provide, so it
    returns None and every shared-policy case in test_dataset_map_num_proc.py skips instead of
    running. _load_trainer_module re-imports the trainer per test, so scoping beats a one-shot
    cleanup after the import below. A real install stubs nothing, so this is a no-op there.
    """
    for name, attrs in _STUB_SPECS:
        _stub_if_missing(name, attrs)
    try:
        yield
    finally:
        while _STUBBED:
            sys.modules.pop(_STUBBED.pop(), None)


with _stubbed():
    from core.training.trainer import UnslothTrainer  # noqa: E402

_preflight = UnslothTrainer._preflight_first_batch
_renders_empty = UnslothTrainer._chat_template_renders_empty
_auto_detect_eval = UnslothTrainer._auto_detect_eval_split_from_hf
_resolve_eval_split = UnslothTrainer._resolve_eval_split_from_dataset
_load_and_format_dataset = UnslothTrainer.load_and_format_dataset


class _FakeInnerTrainer:
    def __init__(
        self,
        *,
        batch = None,
        dataloader_error = None,
        train_dataset = None,
    ):
        self._batch = batch
        self._dataloader_error = dataloader_error
        self.train_dataset = train_dataset

    def get_train_dataloader(self):
        if self._dataloader_error is not None:
            raise self._dataloader_error
        return [self._batch]


def _fake_self(
    *,
    inner,
    model_name = "org/Some-Model",
    tokenizer = None,
):
    s = SimpleNamespace(trainer = inner, model_name = model_name, tokenizer = tokenizer)
    # Bind real methods so self._chat_template_renders_empty() resolves.
    s._preflight_first_batch = _preflight.__get__(s)
    s._chat_template_renders_empty = _renders_empty.__get__(s)
    return s


class _EmptyTemplateTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize = False,
        add_generation_prompt = False,
    ):
        return ""


class _RealTemplateTokenizer:
    def apply_chat_template(
        self,
        messages,
        tokenize = False,
        add_generation_prompt = False,
    ):
        return "<|im_start|>user\nhi<|im_end|>"


class _SizedDataset:
    def __init__(
        self,
        size,
        splits = (),
    ):
        self.size = size
        self.info = SimpleNamespace(splits = {name: object() for name in splits})
        self.shuffle_seeds = []

    def __len__(self):
        return self.size

    def select(self, indices):
        selected = _SizedDataset(len(indices), tuple(self.info.splits))
        selected.shuffle_seeds = list(self.shuffle_seeds)
        return selected

    def shuffle(self, seed = None):
        shuffled = _SizedDataset(self.size, tuple(self.info.splits))
        shuffled.shuffle_seeds = [*self.shuffle_seeds, seed]
        return shuffled


class _SplittableDataset(_SizedDataset):
    def __init__(
        self,
        size,
        calls = None,
    ):
        super().__init__(size)
        self.calls = [] if calls is None else calls

    def train_test_split(self, *, test_size, seed):
        self.calls.append((test_size, seed))
        return {
            "train": _SizedDataset(self.size - test_size),
            "test": _SizedDataset(test_size),
        }


def _dataset_loader_self():
    trainer = SimpleNamespace(
        should_stop = False,
        _audio_type = None,
        is_audio_vlm = False,
        is_vlm = False,
        model_name = "org/model",
        tokenizer = None,
        _update_progress = lambda **kwargs: None,
        _resolve_eval_split_from_dataset = lambda dataset: None,
    )
    trainer._auto_detect_eval_split_from_hf = _auto_detect_eval.__get__(trainer)
    trainer.load_and_format_dataset = _load_and_format_dataset.__get__(trainer)
    return trainer


def _patch_dataset_formatting(monkeypatch):
    monkeypatch.setattr(
        "core.training.trainer.format_and_template_dataset",
        lambda dataset, **kwargs: {
            "dataset": dataset,
            "detected_format": "test",
            "success": True,
        },
    )


def test_cached_auto_eval_uses_supplied_splits_and_loader(monkeypatch):
    remote_calls: list[object] = []
    local_calls: list[str] = []
    expected = _SizedDataset(20)

    def fail_remote(*args, **kwargs):
        remote_calls.append((args, kwargs))
        raise AssertionError("remote dataset access is not allowed")

    monkeypatch.setattr("core.training.trainer.load_dataset", fail_remote)
    monkeypatch.setattr(sys.modules["datasets"], "get_dataset_split_names", fail_remote)

    result = _auto_detect_eval(
        SimpleNamespace(),
        "org/dataset",
        None,
        available_splits = ["train", "validation"],
        split_loader = lambda split: local_calls.append(split) or expected,
        excluded_split = "train",
    )

    assert result is expected
    assert local_calls == ["validation"]
    assert remote_calls == []


def test_cached_auto_eval_with_no_splits_stays_local(monkeypatch):
    remote_calls: list[object] = []

    def fail_remote(*args, **kwargs):
        remote_calls.append((args, kwargs))
        raise AssertionError("remote dataset access is not allowed")

    monkeypatch.setattr("core.training.trainer.load_dataset", fail_remote)
    monkeypatch.setattr(sys.modules["datasets"], "get_dataset_split_names", fail_remote)

    result = _auto_detect_eval(
        SimpleNamespace(),
        "org/dataset",
        None,
        available_splits = [],
        split_loader = fail_remote,
        excluded_split = "train",
    )

    assert result is None
    assert remote_calls == []


def test_cached_auto_eval_excludes_training_split():
    local_calls: list[str] = []

    result = _auto_detect_eval(
        SimpleNamespace(),
        "org/dataset",
        None,
        available_splits = ["train", "validation"],
        split_loader = lambda split: local_calls.append(split) or _SizedDataset(20),
        excluded_split = "validation",
    )

    assert result is None
    assert local_calls == []


def test_cached_auto_eval_excludes_every_split_in_training_instruction():
    local_calls: list[str] = []

    result = _auto_detect_eval(
        SimpleNamespace(),
        "org/dataset",
        None,
        available_splits = ["train", "validation", "test"],
        split_loader = lambda split: local_calls.append(split) or _SizedDataset(20),
        excluded_split = ("train", "validation"),
    )

    assert isinstance(result, _SizedDataset)
    assert local_calls == ["test"]


def test_cached_auto_eval_propagates_loader_failure_for_exact_resume():
    def fail_load(_split):
        raise FileNotFoundError("validation")

    with pytest.raises(FileNotFoundError, match = "validation"):
        _auto_detect_eval(
            SimpleNamespace(),
            "org/dataset",
            None,
            available_splits = ["train", "validation"],
            split_loader = fail_load,
            excluded_split = "train",
            strict_split_loading = True,
        )


def test_auto_eval_probe_failure_records_a_durable_warning(monkeypatch):
    warnings: list[str] = []

    def fail_probe(**_kwargs):
        raise OSError("metadata unavailable")

    monkeypatch.setattr(sys.modules["datasets"], "get_dataset_split_names", fail_probe)

    result = _auto_detect_eval(
        SimpleNamespace(_record_warning = warnings.append),
        "org/dataset",
        None,
    )

    assert result is None
    assert len(warnings) == 1
    assert "held-out split" in warnings[0]
    assert "metadata unavailable" in warnings[0]


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, False),
        (-1, False),
        (False, False),
        (True, False),
        (None, False),
        ("", False),
        ("not-a-number", False),
        (float("nan"), False),
        (float("inf"), False),
        (0.1, True),
        ("2", True),
    ],
)
def test_evaluation_enabled_accepts_only_finite_positive_intervals(value, expected):
    from core.training.eval_dataset import evaluation_enabled
    assert evaluation_enabled(value) is expected


@pytest.mark.parametrize(
    "rows, expected_eval_rows",
    [
        (0, None),
        (31, None),
        (32, 16),
        (1_000, 50),
        (10_000, 128),
    ],
)
def test_shared_eval_split_is_bounded_and_deterministic(rows, expected_eval_rows):
    from core.training.eval_dataset import split_dataset_for_evaluation

    dataset = _SplittableDataset(rows)
    result = split_dataset_for_evaluation(dataset)

    if expected_eval_rows is None:
        assert result is None
        assert dataset.calls == []
        return

    train, evaluation = result
    assert len(train) == rows - expected_eval_rows
    assert len(evaluation) == expected_eval_rows
    assert dataset.calls == [(expected_eval_rows, 3407)]


def test_torch_eval_split_warns_when_dataset_is_too_small():
    warnings: list[str] = []
    owner = SimpleNamespace(_record_warning = warnings.append)

    result = _resolve_eval_split(owner, _SplittableDataset(31))

    assert result is None
    assert len(warnings) == 1
    assert "only 31 rows" in warnings[0]
    assert "at least 32" in warnings[0]


def test_cached_train_auto_eval_stays_on_pinned_dataset(monkeypatch):
    from hub.utils import dataset_cache

    _patch_dataset_formatting(monkeypatch)
    trainer = _dataset_loader_self()
    cache_calls: list[str] = []
    train = _SizedDataset(40, ("train", "validation"))
    validation = _SizedDataset(20, ("train", "validation"))

    def load_cached(
        repo_id,
        local_path,
        *,
        subset,
        split,
        token = None,
    ):
        cache_calls.append(split)
        return validation if split == "validation" else train

    def fail_remote(*args, **kwargs):
        raise AssertionError("remote dataset access is not allowed")

    monkeypatch.setattr(dataset_cache, "load_cached_hf_dataset", load_cached)
    monkeypatch.setattr("core.training.trainer.load_dataset", fail_remote)
    monkeypatch.setattr(sys.modules["datasets"], "get_dataset_split_names", fail_remote)

    result = trainer.load_and_format_dataset(
        "org/dataset",
        eval_steps = 1,
        dataset_local_files_only = True,
        dataset_local_path = "/cache/snapshot",
    )

    assert result is not None
    assert result[0]["dataset"] is train
    assert result[1] is validation
    assert cache_calls == ["train", "validation"]


def test_bounded_cached_train_forwards_only_required_row_count(monkeypatch):
    from hub.utils import dataset_cache

    _patch_dataset_formatting(monkeypatch)
    trainer = _dataset_loader_self()
    cache_calls: list[tuple[str, int | None]] = []
    validation = _SizedDataset(20, ("train", "validation"))

    def load_cached(
        repo_id,
        local_path,
        *,
        subset,
        split,
        token = None,
        row_limit = None,
    ):
        cache_calls.append((split, row_limit))
        if split == "validation":
            return validation
        return _SizedDataset(row_limit or 100, ("train", "validation"))

    monkeypatch.setattr(dataset_cache, "load_cached_hf_dataset", load_cached)
    monkeypatch.setattr(
        "core.training.trainer.load_dataset",
        lambda *args, **kwargs: pytest.fail("remote dataset access is not allowed"),
    )

    result = trainer.load_and_format_dataset(
        "org/dataset",
        eval_steps = 1,
        dataset_slice_start = 8,
        dataset_slice_end = 32,
        dataset_local_files_only = True,
        dataset_local_path = "/cache/snapshot",
    )

    assert result is not None
    assert len(result[0]["dataset"]) == 25
    assert result[1] is validation
    assert cache_calls == [("train", 33), ("validation", None)]


def _cached_only_loader(
    monkeypatch,
    train,
    validation = None,
):
    """A trainer whose dataset comes from cache, with remote access fatal."""
    from hub.utils import dataset_cache

    _patch_dataset_formatting(monkeypatch)

    # row_limit arrives on the explicit-slice path, which fetches end + 1 rows.
    def load_cached(
        repo_id,
        local_path,
        *,
        subset,
        split,
        token = None,
        row_limit = None,
    ):
        if split == "validation":
            return validation
        return _SizedDataset(row_limit) if row_limit else train

    def fail_remote(*args, **kwargs):
        raise AssertionError("remote dataset access is not allowed")

    monkeypatch.setattr(dataset_cache, "load_cached_hf_dataset", load_cached)
    monkeypatch.setattr("core.training.trainer.load_dataset", fail_remote)
    monkeypatch.setattr(sys.modules["datasets"], "get_dataset_split_names", fail_remote)
    return _dataset_loader_self()


def test_max_steps_dataset_rows_bounds_the_run():
    from core.training.dataset_bounds import (
        MAX_STEPS_ROW_SLACK,
        MIN_MAX_STEPS_ROWS,
        max_steps_dataset_rows,
    )

    # An epoch-bounded run reads its whole dataset.
    assert max_steps_dataset_rows(0, 2, 4) is None
    assert max_steps_dataset_rows(None, 2, 4) is None

    assert max_steps_dataset_rows(2000, 8, 16) == 2000 * 8 * 16 * MAX_STEPS_ROW_SLACK
    # Small runs land on the floor, not a statistically useless handful.
    assert max_steps_dataset_rows(30, 2, 4) == MIN_MAX_STEPS_ROWS
    assert max_steps_dataset_rows(1, 1, 1) == MIN_MAX_STEPS_ROWS


def test_max_steps_bound_subsets_before_formatting(monkeypatch):
    # 30 steps must not tokenize a corpus of 500k rows.
    trainer = _cached_only_loader(monkeypatch, _SizedDataset(500_000))

    result = trainer.load_and_format_dataset(
        "org/dataset",
        dataset_local_files_only = True,
        dataset_local_path = "/cache/snapshot",
        max_train_rows = 1024,
        max_train_rows_seed = 99,
    )

    assert result is not None
    bounded = result[0]["dataset"]
    assert len(bounded) == 1024
    # Shuffled: the head of a corpus ordered by source is not a sample of it.
    assert bounded.shuffle_seeds == [99]


def test_max_steps_bound_leaves_a_small_dataset_alone(monkeypatch):
    train = _SizedDataset(40)
    trainer = _cached_only_loader(monkeypatch, train)

    result = trainer.load_and_format_dataset(
        "org/dataset",
        dataset_local_files_only = True,
        dataset_local_path = "/cache/snapshot",
        max_train_rows = 1024,
    )

    assert result is not None
    # Untouched: no shuffle cost or reordering for a run that reads it all.
    assert result[0]["dataset"] is train


def test_max_steps_bound_defers_to_an_explicit_slice(monkeypatch):
    trainer = _cached_only_loader(monkeypatch, _SizedDataset(500_000))

    result = trainer.load_and_format_dataset(
        "org/dataset",
        dataset_local_files_only = True,
        dataset_local_path = "/cache/snapshot",
        dataset_slice_start = 8,
        dataset_slice_end = 32,
        max_train_rows = 1024,
    )

    assert result is not None
    sliced = result[0]["dataset"]
    # The user named the rows; the bound must not resample them.
    assert len(sliced) == 25
    assert sliced.shuffle_seeds == []


def test_max_steps_bound_defers_to_a_split_instruction(monkeypatch):
    train = _SizedDataset(500_000)
    trainer = _cached_only_loader(monkeypatch, train)

    result = trainer.load_and_format_dataset(
        "org/dataset",
        dataset_local_files_only = True,
        dataset_local_path = "/cache/snapshot",
        train_split = "train[1000:200000]",
        max_train_rows = 1024,
    )

    assert result is not None
    # A bracketed split names rows like the numeric slice fields do.
    assert result[0]["dataset"] is train


def test_max_steps_bound_is_off_without_it(monkeypatch):
    train = _SizedDataset(500_000)
    trainer = _cached_only_loader(monkeypatch, train)

    result = trainer.load_and_format_dataset(
        "org/dataset",
        dataset_local_files_only = True,
        dataset_local_path = "/cache/snapshot",
    )

    assert result is not None
    assert result[0]["dataset"] is train


def test_max_steps_dataset_rows_survives_unusable_numbers():
    from core.training.dataset_bounds import MIN_MAX_STEPS_ROWS, max_steps_dataset_rows

    # The DB and direct callers hand over Nones and strings; this must never raise.
    assert max_steps_dataset_rows(30, None, None) == MIN_MAX_STEPS_ROWS
    assert max_steps_dataset_rows(30, "2", "4") == MIN_MAX_STEPS_ROWS
    assert max_steps_dataset_rows("30", 2, 4) == MIN_MAX_STEPS_ROWS
    assert max_steps_dataset_rows(-5, 2, 4) is None
    assert max_steps_dataset_rows("not a number", 2, 4) is None
    # A bound past any corpus is a no-op at the apply site, not an error.
    assert max_steps_dataset_rows(10**9, 2, 4) == 10**9 * 8 * 4


def _single_process_launch(monkeypatch):
    """Clear every launcher variable, so a bound reads as Unsloth's own launch."""
    from core.training.dataset_bounds import WORLD_SIZE_ENV_FILES, WORLD_SIZE_ENV_VARS
    for name in WORLD_SIZE_ENV_VARS + WORLD_SIZE_ENV_FILES:
        monkeypatch.delenv(name, raising = False)


def test_max_steps_dataset_rows_scales_with_world_size(monkeypatch):
    from core.training.dataset_bounds import (
        MAX_STEPS_ROW_SLACK,
        MIN_MAX_STEPS_ROWS,
        max_steps_dataset_rows,
    )

    _single_process_launch(monkeypatch)

    # One process is what the bound has always assumed: identical to omitting it.
    assert max_steps_dataset_rows(2000, 8, 16, world_size = 1) == max_steps_dataset_rows(2000, 8, 16)
    assert max_steps_dataset_rows(2000, 8, 16) == 2000 * 8 * 16 * MAX_STEPS_ROW_SLACK
    assert max_steps_dataset_rows(30, 2, 4, world_size = 1) == MIN_MAX_STEPS_ROWS

    # Every replica draws its own batch per step, so the subset grows with them.
    for world_size in (2, 4, 8):
        assert (
            max_steps_dataset_rows(2000, 8, 16, world_size = world_size)
            == 2000 * 8 * 16 * world_size * MAX_STEPS_ROW_SLACK
        )
        # The slack is what stops a run recycling rows, so it must survive the scaling.
        rows = max_steps_dataset_rows(60, 2, 4, world_size = world_size)
        assert rows >= 60 * 2 * 4 * world_size * MAX_STEPS_ROW_SLACK

    # An unbounded run stays unbounded however many replicas read it.
    assert max_steps_dataset_rows(0, 2, 4, world_size = 8) is None


def test_max_steps_dataset_rows_survives_an_unusable_world_size(monkeypatch):
    from core.training.dataset_bounds import MIN_MAX_STEPS_ROWS, max_steps_dataset_rows

    _single_process_launch(monkeypatch)

    # A launcher that reports nothing, or nonsense, must read as one process rather
    # than raise or collapse the subset to nothing.
    baseline = max_steps_dataset_rows(2000, 8, 16)
    for world_size in (None, 0, -4, "", "auto", "not a number", float("inf"), object()):
        assert max_steps_dataset_rows(2000, 8, 16, world_size = world_size) == baseline
    assert max_steps_dataset_rows(30, 2, 4, world_size = None) == MIN_MAX_STEPS_ROWS

    # A string count is what an env carries, and it still has to scale.
    assert max_steps_dataset_rows(2000, 8, 16, world_size = "4") == baseline * 4


def test_world_size_comes_from_the_launcher_env(monkeypatch):
    from core.training.dataset_bounds import (
        MAX_STEPS_ROW_SLACK,
        max_steps_dataset_rows,
        world_size_from_env,
    )

    _single_process_launch(monkeypatch)
    assert world_size_from_env() == 1

    # torchrun and accelerate set WORLD_SIZE; mlx.launch and mpirun set their own.
    for name, size in (("WORLD_SIZE", 4), ("MLX_WORLD_SIZE", 2), ("OMPI_COMM_WORLD_SIZE", 8)):
        _single_process_launch(monkeypatch)
        monkeypatch.setenv(name, str(size))
        assert world_size_from_env() == size
        assert max_steps_dataset_rows(2000, 8, 16) == 2000 * 8 * 16 * size * MAX_STEPS_ROW_SLACK

    # A multi-node torchrun sets both, and the global count is the one that sizes rows.
    _single_process_launch(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    assert world_size_from_env() == 16

    # Junk in the env reads as a single process, not as a crash.
    for junk in ("", "auto", "0", "-2", "3.5"):
        _single_process_launch(monkeypatch)
        monkeypatch.setenv("WORLD_SIZE", junk)
        assert world_size_from_env() == 1

    # An explicit count is the caller's own detection and outranks the env, which
    # cannot see the visible CUDA devices DataParallel would also split a batch over.
    _single_process_launch(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert max_steps_dataset_rows(2000, 8, 16, world_size = 8) == 2000 * 8 * 16 * 8 * (
        MAX_STEPS_ROW_SLACK
    )
    # A mapping can be passed instead of the process env.
    assert world_size_from_env({"WORLD_SIZE": "4"}) == 4
    assert world_size_from_env({}) == 1


def test_world_size_comes_from_an_mlx_launch_hostfile(tmp_path, monkeypatch):
    """An Apple silicon mlx.launch advertises its ranks as a file, not a number.

    Of mlx.launch's four backends only NCCL (CUDA) exports MLX_WORLD_SIZE. Ring and
    JACCL -- everything a Mac runs -- export a path to a JSON file with one entry per
    rank, so reading only the numeric variables sizes a four-way launch as one process
    and the run re-reads rows it has already trained on.
    """
    from core.training.dataset_bounds import (
        MAX_STEPS_ROW_SLACK,
        max_steps_dataset_rows,
        world_size_from_env,
        world_size_from_rank_files,
    )

    _single_process_launch(monkeypatch)
    assert world_size_from_rank_files() == 1

    # The ring backend writes one "ip:port" list per rank; mlx.launch --hostfile ring-4.
    ring = tmp_path / "ring.json"
    ring.write_text(json.dumps([[f"10.0.0.{rank}:5000"] for rank in range(4)]), encoding = "utf-8")
    _single_process_launch(monkeypatch)
    monkeypatch.setenv("MLX_RANK", "0")
    monkeypatch.setenv("MLX_HOSTFILE", str(ring))
    assert world_size_from_env() == 4
    assert max_steps_dataset_rows(60, 2, 4) == 60 * 2 * 4 * 4 * MAX_STEPS_ROW_SLACK

    # The jaccl backend writes the RDMA matrix, one row per rank.
    rdma = tmp_path / "rdma.json"
    rdma.write_text(
        json.dumps([[None if a == b else "mlx5_0" for b in range(3)] for a in range(3)]),
        encoding = "utf-8",
    )
    _single_process_launch(monkeypatch)
    monkeypatch.setenv("MLX_IBV_DEVICES", str(rdma))
    assert world_size_from_env() == 3

    # A single host is not a distributed launch: mlx.launch writes an empty hostfile.
    empty = tmp_path / "empty.json"
    empty.write_text("", encoding = "utf-8")
    _single_process_launch(monkeypatch)
    monkeypatch.setenv("MLX_HOSTFILE", str(empty))
    assert world_size_from_env() == 1

    # The payload can also be inline, which is how unsloth_cli/_inference.py's
    # _json_rank_count_from_env reads these two, including the {"hosts": [...]} form.
    for inline, expected in (
        (json.dumps([[f"10.0.0.{rank}:5000"] for rank in range(6)]), 6),
        (json.dumps({"hosts": ["a", "b", "c"]}), 3),
        ("  " + json.dumps([["a"], ["b"]]), 2),
        (json.dumps([]), 1),
    ):
        _single_process_launch(monkeypatch)
        monkeypatch.setenv("MLX_HOSTFILE", inline)
        assert world_size_from_env() == expected

    # Nothing about a hostfile may fail a run: unreadable, not JSON, not a list.
    bad_json = tmp_path / "bad.json"
    bad_json.write_text('[["10.0.0.1:5000"],', encoding = "utf-8")
    not_a_list = tmp_path / "object.json"
    not_a_list.write_text(json.dumps({"hosts": 4}), encoding = "utf-8")
    a_directory = tmp_path / "adir"
    a_directory.mkdir()
    for value in (
        str(tmp_path / "missing.json"),
        str(bad_json),
        str(not_a_list),
        str(a_directory),
        '[["10.0.0.1:5000"],',  # inline and truncated
        '{"hosts": 4}',
        "[",
        "{",
        "",
        "   ",
    ):
        _single_process_launch(monkeypatch)
        monkeypatch.setenv("MLX_HOSTFILE", value)
        assert world_size_from_env() == 1

    # A fifo would block open() forever, so only regular files are read at all.
    fifo = tmp_path / "fifo"
    try:
        os.mkfifo(fifo)
    except (AttributeError, NotImplementedError, OSError):
        pass  # Windows has no fifos, which is the point.
    else:
        assert world_size_from_rank_files({"MLX_HOSTFILE": str(fifo)}) == 1

    # A mapping works the same way, and the largest count still wins.
    assert world_size_from_rank_files({"MLX_HOSTFILE": str(ring)}) == 4
    assert world_size_from_env({"MLX_HOSTFILE": str(ring), "WORLD_SIZE": "8"}) == 8
    assert world_size_from_env({"MLX_HOSTFILE": str(ring), "WORLD_SIZE": "2"}) == 4
    assert world_size_from_rank_files({"MLX_HOSTFILE": None}) == 1
    assert world_size_from_rank_files({"MLX_HOSTFILE": 17}) == 1
    assert world_size_from_rank_files({}) == 1


def test_a_rank_file_read_is_capped_in_bytes_not_characters(tmp_path):
    """The cap has to bound what comes off the disk, whatever the file holds.

    A text-mode ``read(n)`` counts CHARACTERS, so a file of 4-byte codepoints
    would pull four times ``MAX_WORLD_SIZE_FILE_BYTES`` into memory on a variable
    that names an arbitrary path. Read in binary and the constant means what it
    says; ``json.loads`` takes bytes, and non-UTF-8 raises ``UnicodeDecodeError``,
    which is a ``ValueError`` and already discarded.
    """
    from core.training.dataset_bounds import (
        MAX_WORLD_SIZE_FILE_BYTES,
        world_size_from_rank_files,
    )

    # Sized so the readings disagree: under the cap in characters, over it in bytes.
    # A text handle reads it whole and answers 8; binary truncates to one process,
    # the safe direction, and reading the whole file is what the cap forbids.
    wide = tmp_path / "wide.json"
    filler = "\U0001f600" * (MAX_WORLD_SIZE_FILE_BYTES // 3)  # 4 bytes per character
    hosts = [filler] + [f"10.0.0.{rank}:5000" for rank in range(7)]
    # ensure_ascii would escape the codepoints back to ASCII and make the two
    # readings agree, which is what this test needs them not to do.
    wide.write_text(json.dumps(hosts, ensure_ascii = False), encoding = "utf-8")
    assert len(wide.read_text(encoding = "utf-8")) < MAX_WORLD_SIZE_FILE_BYTES
    assert wide.stat().st_size > MAX_WORLD_SIZE_FILE_BYTES
    assert world_size_from_rank_files({"MLX_HOSTFILE": str(wide)}) == 1

    # Non-UTF-8 bytes must be discarded, not raised.
    invalid = tmp_path / "invalid.bin"
    invalid.write_bytes(b'["\xff\xfe10.0.0.1:5000"]')
    assert world_size_from_rank_files({"MLX_HOSTFILE": str(invalid)}) == 1


def test_the_launcher_env_report_names_the_variable_that_claimed_the_ranks():
    """A stale size variable is otherwise invisible.

    The report exists so a user whose single-machine run is told it makes several
    passes can see which variable said so. It must never raise and never grow
    without bound: MLX_HOSTFILE legitimately carries a whole JSON payload.
    """
    from core.training.dataset_bounds import world_size_env_report

    assert world_size_env_report({}) == "no launcher variable set"
    assert world_size_env_report({"WORLD_SIZE": ""}) == "no launcher variable set"
    assert world_size_env_report({"NOT_A_LAUNCHER": "8"}) == "no launcher variable set"

    report = world_size_env_report({"OMPI_COMM_WORLD_SIZE": "8", "WORLD_SIZE": "2"})
    assert "OMPI_COMM_WORLD_SIZE=8" in report and "WORLD_SIZE=2" in report

    long_payload = world_size_env_report({"MLX_HOSTFILE": json.dumps(["h"] * 500)})
    assert len(long_payload) < 200

    class _Hostile:
        def get(self, name):
            raise RuntimeError("environment lookup exploded")

    assert world_size_env_report(_Hostile()) == "no launcher variable set"


def test_effective_packing_decides_the_opt_out():
    from core.training.dataset_bounds import effective_packing, max_train_rows_for_config

    text = {"max_steps": 30, "batch_size": 2, "gradient_accumulation_steps": 4}

    # Packing spans unknown rows per sample, so text runs opt out.
    assert effective_packing({**text, "packing": True}) is True
    assert max_train_rows_for_config({**text, "packing": True}) is None

    # A caller that probed a never-packing branch keeps the bound despite the flag.
    assert effective_packing({**text, "packing": True}, branch_never_packs = True) is False
    assert max_train_rows_for_config({**text, "packing": True}, branch_never_packs = True) == 1024

    # The dataset flags establish nothing: client-supplied and true on a column-NAME
    # match, so a text model with an "audio" column still trains on the packing path.
    assert effective_packing({**text, "packing": True, "is_dataset_image": True}) is True
    assert effective_packing({**text, "packing": True, "is_dataset_audio": True}) is True
    assert max_train_rows_for_config({**text, "packing": True, "is_dataset_audio": True}) is None

    # An epoch-bounded run is unbounded whatever packing says.
    assert max_train_rows_for_config({"max_steps": 0, "packing": False}) is None

    # Raw-text and CPT do not enter into it here: the caller decides the branch,
    # since vision is gated on `not raw_text_mode` while audio holds either way.
    for raw in ({"training_type": "Continued Pretraining"}, {"format_type": "raw"}):
        assert effective_packing({**text, **raw, "packing": True}, branch_never_packs = True) is False
        assert effective_packing({**text, **raw, "packing": True}) is True
        # Without packing they bound like anything else.
        assert effective_packing({**text, **raw}, branch_never_packs = True) is False
        assert max_train_rows_for_config({**text, **raw}, branch_never_packs = True) == 1024


def test_bound_dataset_rows_edges():
    from core.training.dataset_bounds import bound_dataset_rows

    class _Streaming:
        """No __len__, like an IterableDataset."""

        def shuffle(self, seed = None):
            raise AssertionError("a streaming dataset must not be shuffled eagerly")

    exact = _SizedDataset(1024)
    assert bound_dataset_rows(exact, 1024, 3407) is exact
    assert len(bound_dataset_rows(_SizedDataset(1025), 1024, 3407)) == 1024

    # A non-positive bound would otherwise select an empty dataset.
    untouched = _SizedDataset(500_000)
    assert bound_dataset_rows(untouched, 0, 3407) is untouched
    assert bound_dataset_rows(untouched, -5, 3407) is untouched
    assert bound_dataset_rows(untouched, None, 3407) is untouched

    streaming = _Streaming()
    assert bound_dataset_rows(streaming, 1024, 3407) is streaming

    # An uncoercible seed still has to produce a subset.
    assert len(bound_dataset_rows(_SizedDataset(500_000), 1024, None)) == 1024


def test_bound_dataset_rows_keeps_seed_zero():
    from datasets import Dataset

    from core.training.dataset_bounds import bound_dataset_rows

    source = Dataset.from_dict({"row": list(range(5000))})

    # 0 is a legitimate seed and must not collapse onto the default.
    assert (
        bound_dataset_rows(source, 1024, 0)["row"] != bound_dataset_rows(source, 1024, 3407)["row"]
    )
    assert bound_dataset_rows(source, 1024, 0)["row"] == bound_dataset_rows(source, 1024, 0)["row"]


def test_bound_dataset_rows_survives_a_hostile_seed():
    from datasets import Dataset

    from core.training.dataset_bounds import bound_dataset_rows

    source = Dataset.from_dict({"row": list(range(3000))})

    # numpy rejects negative seeds and -1 is a common sentinel; json accepts
    # Infinity, so a stored config can hold one. Neither may take a run down.
    for seed in (-1, -3407, float("inf"), float("nan"), "3407", None, "seed"):
        assert len(bound_dataset_rows(source, 1024, seed)) == 1024


def test_max_steps_dataset_rows_survives_infinity():
    from core.training.dataset_bounds import MIN_MAX_STEPS_ROWS, max_steps_dataset_rows

    infinity = float("inf")
    assert max_steps_dataset_rows(infinity, 2, 4) is None
    assert max_steps_dataset_rows(30, infinity, 4) == MIN_MAX_STEPS_ROWS
    assert max_steps_dataset_rows(30, 2, infinity) == MIN_MAX_STEPS_ROWS


def test_bound_dataset_rows_leaves_a_dataset_dict_alone():
    from datasets import Dataset, DatasetDict

    from core.training.dataset_bounds import bound_dataset_rows

    # len() on a DatasetDict is the split count, and it has no select().
    splits = DatasetDict(
        {
            "train": Dataset.from_dict({"row": list(range(5000))}),
            "test": Dataset.from_dict({"row": list(range(100))}),
        }
    )
    assert bound_dataset_rows(splits, 1024, 3407) is splits


def test_row_bound_marker_round_trips_through_a_resume(tmp_path):
    from core.training.dataset_bounds import record_row_bound, row_bound_for_resume

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "checkpoint-30"
    checkpoint.mkdir()

    # Not resuming: the freshly computed pair.
    assert row_bound_for_resume(None, 4096, 3407) == (4096, 3407)

    record_row_bound(str(run_dir), 4096, 3407)
    # Resuming reads back the original bound, so edits to max_steps or batch size
    # do not move the rows or their order.
    assert row_bound_for_resume(str(checkpoint), 40960, 99) == (4096, 3407)
    # The run directory is accepted as well as a checkpoint inside it.
    assert row_bound_for_resume(str(run_dir), 40960, 99) == (4096, 3407)

    # A run that was never bounded stays unbounded on resume.
    unbounded = tmp_path / "unbounded"
    unbounded.mkdir()
    record_row_bound(str(unbounded), None, 3407)
    assert row_bound_for_resume(str(unbounded / "checkpoint-5"), 1024, 3407) == (None, 3407)


def test_row_bound_marker_survives_a_run_directory_named_like_a_checkpoint(tmp_path):
    from core.training.dataset_bounds import record_row_bound, row_bound_for_resume

    # A run directory whose name merely starts with the checkpoint prefix is not a
    # checkpoint; taking its parent would file the marker one level too high.
    run_dir = tmp_path / "checkpoint-model__project-x"
    (run_dir / "checkpoint-30").mkdir(parents = True)
    record_row_bound(str(run_dir), 4096, 3407)

    assert (run_dir / "unsloth_row_bound.json").exists()
    assert not (tmp_path / "unsloth_row_bound.json").exists()
    assert row_bound_for_resume(str(run_dir / "checkpoint-30"), 40960, 99) == (4096, 3407)


def test_row_bound_marker_is_replaced_atomically(tmp_path):
    import os

    from core.training.dataset_bounds import record_row_bound, row_bound_for_resume

    # A resume rewrites an already valid marker: truncating in place then failing
    # would leave an empty file, read as "no marker".
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record_row_bound(str(run_dir), 4096, 3407)

    real_replace = os.replace

    def _fail_replace(src, dst):
        raise OSError(28, "No space left on device")

    os.replace = _fail_replace
    try:
        record_row_bound(str(run_dir), 8192, 99)
    finally:
        os.replace = real_replace

    assert row_bound_for_resume(str(run_dir), 40960, 99) == (4096, 3407)
    # And the temporary file is cleaned up.
    assert [p.name for p in run_dir.iterdir()] == ["unsloth_row_bound.json"]


def test_run_dir_for_a_bare_relative_checkpoint(tmp_path, monkeypatch):
    from core.training.dataset_bounds import (
        record_row_bound,
        row_bound_for_resume,
        run_dir_for_checkpoint,
    )

    # "checkpoint-30" splits to an empty head; its run dir is the cwd, not itself,
    # or the marker is looked for one level too deep and the run reads as legacy.
    assert run_dir_for_checkpoint("checkpoint-30") == os.curdir
    assert run_dir_for_checkpoint("run/checkpoint-30") == "run"
    # A relative run directory is still itself.
    assert run_dir_for_checkpoint("checkpoint-model") == "checkpoint-model"

    run_dir = tmp_path / "run"
    (run_dir / "checkpoint-30").mkdir(parents = True)
    record_row_bound(str(run_dir), 4096, 3407)
    monkeypatch.chdir(run_dir)
    assert row_bound_for_resume("checkpoint-30", 40960, 99) == (4096, 3407)


def test_record_row_bound_reports_whether_it_wrote():
    from core.training.dataset_bounds import record_row_bound

    # The caller logs a failure rather than failing the run: the dataset is already
    # bounded by now, so there is nothing to fall back to.
    assert record_row_bound(None, 1024, 3407) is False
    assert record_row_bound("/definitely/not/a/directory/here", 1024, 3407) is False


def test_row_bound_is_dropped_for_a_checkpoint_that_predates_it(tmp_path):
    from core.training.dataset_bounds import record_row_bound, row_bound_for_resume

    # A checkpoint written before the marker trained on the whole corpus in natural
    # order. Both trainers resume by batch index, so a subset would continue on
    # unrelated rows: no bound, whatever its size.
    legacy = tmp_path / "legacy"
    (legacy / "checkpoint-30").mkdir(parents = True)
    assert row_bound_for_resume(str(legacy / "checkpoint-30"), 1024, 3407) == (None, 3407)

    # Including the range an arithmetic estimate could not tell apart.
    (legacy / "checkpoint-30" / "trainer_state.json").write_text(
        json.dumps({"global_step": 15, "epoch": 120 / 1500, "train_batch_size": 2})
    )
    assert row_bound_for_resume(str(legacy / "checkpoint-30"), 1024, 3407) == (None, 3407)

    # An unreadable or truncated marker reads as legacy, never as a bound.
    for name, body in (("empty", "{}"), ("broken", "not json"), ("null", "null")):
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "unsloth_row_bound.json").write_text(body)
        assert row_bound_for_resume(str(run_dir), 1024, 3407) == (None, 3407)

    # An unwritable marker leaves the resume unbounded rather than failing the run.
    record_row_bound(str(tmp_path / "does" / "not" / "exist"), 1024, 3407)
    record_row_bound(None, 1024, 3407)


def test_bound_dataset_rows_is_deterministic_and_seed_sensitive():
    from datasets import Dataset

    from core.training.dataset_bounds import bound_dataset_rows

    source = Dataset.from_dict({"row": list(range(5000)), "text": [f"t{i}" for i in range(5000)]})

    first = bound_dataset_rows(source, 1024, 3407)["row"]
    second = bound_dataset_rows(source, 1024, 3407)["row"]
    other = bound_dataset_rows(source, 1024, 99)["row"]

    assert len(first) == 1024
    assert first == second
    assert first != other
    # The head of a corpus ordered by source is not a sample of it.
    assert first != list(range(1024))
    # Features survive shuffle+select, so the formatting passes still work.
    assert bound_dataset_rows(source, 1024, 3407).column_names == ["row", "text"]


def test_bound_leaves_enough_rows_after_the_eval_carve():
    from datasets import Dataset

    from core.training.dataset_bounds import bound_dataset_rows, max_train_rows_for_config
    from core.training.eval_dataset import split_dataset_for_evaluation

    config = {"max_steps": 30, "batch_size": 2, "gradient_accumulation_steps": 4}
    rows = max_train_rows_for_config(config)
    source = Dataset.from_dict({"text": [f"t{i}" for i in range(500_000)]})

    bounded = bound_dataset_rows(source, rows, 3407)
    train, _eval = split_dataset_for_evaluation(bounded)

    # The eval carve is what MAX_STEPS_ROW_SLACK budgets for: the run must still
    # reach max_steps without re-reading rows.
    needed = config["max_steps"] * config["batch_size"] * config["gradient_accumulation_steps"]
    assert len(train) >= needed


def test_bound_leaves_enough_rows_for_every_rank_after_the_eval_carve(monkeypatch):
    from datasets import Dataset

    from core.training.dataset_bounds import bound_dataset_rows, max_train_rows_for_config
    from core.training.eval_dataset import split_dataset_for_evaluation

    _single_process_launch(monkeypatch)

    config = {"max_steps": 60, "batch_size": 2, "gradient_accumulation_steps": 4}
    source = Dataset.from_dict({"text": [f"t{i}" for i in range(500_000)]})

    for world_size in (1, 2, 4, 8):
        rows = max_train_rows_for_config(config, world_size = world_size)
        bounded = bound_dataset_rows(source, rows, 3407)
        train, _eval = split_dataset_for_evaluation(bounded)

        # Each rank draws its own batch, so the corpus a step consumes is the batch
        # times the ranks; without the factor the eval carve alone tips ws=4 into
        # re-reading rows it has already trained on.
        needed = (
            config["max_steps"]
            * config["batch_size"]
            * config["gradient_accumulation_steps"]
            * world_size
        )
        assert len(train) >= needed

    # Packing still opts out, whatever the launch looks like.
    assert max_train_rows_for_config({**config, "packing": True}, world_size = 8) is None


def test_row_bound_marker_round_trips_a_world_size_scaled_bound(tmp_path, monkeypatch):
    from core.training.dataset_bounds import (
        max_train_rows_for_config,
        record_row_bound,
        row_bound_for_resume,
    )

    _single_process_launch(monkeypatch)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "checkpoint-60"
    checkpoint.mkdir()

    config = {"max_steps": 60, "batch_size": 2, "gradient_accumulation_steps": 4}
    rows = max_train_rows_for_config(config, world_size = 4)
    assert rows == 60 * 2 * 4 * 4 * 4
    assert record_row_bound(str(run_dir), rows, 3407) is True

    # The marker records rows, not the launch that sized them, so a resume on a
    # differently sized machine still trains on the rows the run started with.
    assert row_bound_for_resume(str(checkpoint), max_train_rows_for_config(config), 99) == (
        rows,
        3407,
    )

    # A marker written before this change carries a single-process bound and is
    # still read back verbatim, rather than being rescaled under the run.
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    (legacy_dir / "checkpoint-60").mkdir()
    legacy_rows = 60 * 2 * 4 * 4
    record_row_bound(str(legacy_dir), legacy_rows, 3407)
    assert row_bound_for_resume(str(legacy_dir / "checkpoint-60"), rows, 3407) == (
        legacy_rows,
        3407,
    )

    # And a checkpoint with no marker at all stays unbounded.
    unmarked = tmp_path / "unmarked"
    (unmarked / "checkpoint-60").mkdir(parents = True)
    assert row_bound_for_resume(str(unmarked / "checkpoint-60"), rows, 3407) == (None, 3407)


def test_both_loaders_size_the_bound_for_the_world():
    """The factor is only worth having if both loaders actually pass it.

    Read from source for the same reason as the wiring test below: neither call
    site is reachable without a GPU or Apple hardware.
    """
    import ast
    from pathlib import Path

    worker_src = (Path(__file__).resolve().parents[1] / "core/training/worker.py").read_text(
        encoding = "utf-8"
    )
    tree = ast.parse(worker_src)
    calls = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        calls[node.name] = {
            sub.func.id
            for sub in ast.walk(node)
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
        }

    for loader in ("run_training_process", "_run_mlx_training"):
        assert "_data_parallel_world_size" in calls[loader]
    assert worker_src.count("world_size = _data_parallel_world_size()") == 2
    # The count is a property of this launch, so it must never be read back out of a
    # config that was built on the parent and shipped across a spawn.
    assert 'config.get("world_size")' not in worker_src


def test_data_parallel_world_size_counts_ranks_and_devices(monkeypatch):
    import types

    from core.training import worker as training_worker

    _single_process_launch(monkeypatch)

    # No torch in sys.modules is the MLX host: the env is all there is.
    monkeypatch.setitem(sys.modules, "torch", None)
    assert training_worker._data_parallel_world_size() == 1
    monkeypatch.setenv("MLX_WORLD_SIZE", "4")
    assert training_worker._data_parallel_world_size() == 4

    def _torch(devices, world_size = None):
        distributed = types.SimpleNamespace(
            is_available = lambda: world_size is not None,
            is_initialized = lambda: world_size is not None,
            get_world_size = lambda: world_size,
        )
        return types.SimpleNamespace(
            cuda = types.SimpleNamespace(device_count = lambda: devices),
            distributed = distributed,
        )

    _single_process_launch(monkeypatch)
    # One visible GPU and no launcher is today's single-process run, unchanged.
    monkeypatch.setitem(sys.modules, "torch", _torch(1))
    assert training_worker._data_parallel_world_size() == 1

    # transformers wraps a non-distributed multi-GPU run in DataParallel and scales
    # the train batch by the visible device count, which no env reports.
    monkeypatch.setitem(sys.modules, "torch", _torch(4))
    assert training_worker._data_parallel_world_size() == 4

    # A torchrun rank sees the whole node but trains on its own shard: the larger of
    # the two, never the product, since a distributed run pins n_gpu to 1.
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setitem(sys.modules, "torch", _torch(8, world_size = 8))
    assert training_worker._data_parallel_world_size() == 8

    # CPU-only and a torch whose CUDA probe raises both read as one process.
    _single_process_launch(monkeypatch)
    monkeypatch.setitem(sys.modules, "torch", _torch(0))
    assert training_worker._data_parallel_world_size() == 1

    def _raises():
        raise RuntimeError("no CUDA driver")

    broken = types.SimpleNamespace(
        cuda = types.SimpleNamespace(device_count = _raises),
        distributed = types.SimpleNamespace(is_available = _raises, is_initialized = _raises),
    )
    monkeypatch.setitem(sys.modules, "torch", broken)
    assert training_worker._data_parallel_world_size() == 1


def test_both_loaders_apply_the_row_bound():
    """Guards the wiring: the helpers are useless if a loader stops calling them.

    Read from source: the CUDA worker needs a GPU and the MLX one Apple hardware,
    so neither call site is otherwise reachable in CI.
    """
    import ast
    from pathlib import Path

    worker_src = (Path(__file__).resolve().parents[1] / "core/training/worker.py").read_text(
        encoding = "utf-8"
    )
    tree = ast.parse(worker_src)
    calls = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        names = {
            sub.func.id
            for sub in ast.walk(node)
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
        }
        calls[node.name] = names

    # The CUDA worker derives the bound and hands it to load_and_format_dataset.
    assert "max_train_rows_for_config" in calls["run_training_process"]
    assert "max_train_rows = max_train_rows" in worker_src
    # The MLX worker loads its own dataset, so it bounds its own rows.
    assert "bound_dataset_rows" in calls["_slice"]
    assert "max_train_rows_for_config" in calls["_run_mlx_training"]
    # Both must resume on the recorded bound and record their own, or a resume
    # silently trains on rows the checkpoint never saw.
    for loader in ("run_training_process", "_run_mlx_training"):
        assert "row_bound_for_resume" in calls[loader]
        assert "record_row_bound" in calls[loader]

    # Both pass the detected branch rather than defaulting: the client-supplied
    # dataset flags cannot stand in for it.
    assert worker_src.count("branch_never_packs = ") >= 2
    # And the CUDA one computes it only after the model probe has set it.
    assert worker_src.index("_pre_detect_training_model(\n") < worker_src.index(
        "branch_never_packs = bool("
    )


def test_mlx_adapter_keeps_one_source_of_truth_for_the_bound():
    from core.training.training import _build_training_worker_config

    config = _build_training_worker_config(
        {"model_name": "org/model", "max_steps": 30, "batch_size": 2}
    )
    # The normalized config is a whitelist: a forwarded copy of the bound would be
    # dropped here and silently disagree with what the worker computes.
    assert "max_train_rows" not in config
    assert "max_train_rows_seed" not in config
    # Everything the worker needs to recompute it does survive.
    assert config["max_steps"] == 30
    assert config["batch_size"] == 2
    assert config["gradient_accumulation_steps"] == 4
    assert config["random_seed"] == 3407


def test_remote_train_fallback_keeps_auto_eval_remote(monkeypatch):
    from hub.utils import dataset_cache

    _patch_dataset_formatting(monkeypatch)
    trainer = _dataset_loader_self()
    cache_calls: list[str] = []
    remote_calls: list[tuple[str, str | None]] = []
    train = _SizedDataset(40, ("train", "validation"))
    validation = _SizedDataset(20, ("train", "validation"))

    def load_cached(
        repo_id,
        local_path,
        *,
        subset,
        split,
        token = None,
    ):
        cache_calls.append(split)
        raise FileNotFoundError(split)

    def load_remote(*, path, split, **kwargs):
        remote_calls.append((split, kwargs.get("revision")))
        return validation if split == "validation" else train

    monkeypatch.setattr(dataset_cache, "load_cached_hf_dataset", load_cached)
    monkeypatch.setattr("core.training.trainer.load_dataset", load_remote)
    monkeypatch.setattr(
        sys.modules["datasets"],
        "get_dataset_split_names",
        lambda **kwargs: ["train", "validation"],
    )

    result = trainer.load_and_format_dataset(
        "org/dataset",
        eval_steps = 1,
        dataset_local_files_only = True,
        dataset_local_path = "/cache/snapshot",
        dataset_revision = "dataset-commit",
    )

    assert result is not None
    assert result[0]["dataset"] is train
    assert result[1] is validation
    assert cache_calls == ["train"]
    assert remote_calls == [("train", "dataset-commit"), ("validation", "dataset-commit")]


def test_first_remote_train_load_records_exact_dataset_snapshot(monkeypatch, tmp_path):
    from hub.utils import hf_cache_state

    _patch_dataset_formatting(monkeypatch)
    trainer = _dataset_loader_self()
    snapshot = tmp_path / "datasets--org--dataset" / "snapshots" / "dataset-commit"
    snapshot.mkdir(parents = True)
    (snapshot / "train.parquet").write_bytes(b"dataset")
    train = _SizedDataset(40)
    train.info.download_checksums = {
        "hf://datasets/org/dataset@dataset-commit/train.parquet": {
            "num_bytes": 7,
            "checksum": None,
        }
    }

    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda **kw: [tmp_path])
    monkeypatch.setattr(
        "core.training.trainer.load_dataset",
        lambda **_kwargs: train,
    )

    result = trainer.load_and_format_dataset("org/dataset")

    assert result is not None
    assert trainer.dataset_snapshot_path == str(snapshot.resolve())
    assert trainer.dataset_loaded_from_exact_snapshot is True


def test_manual_eager_slice_attests_original_hub_stream(monkeypatch, tmp_path):
    from hub.utils import hf_cache_state

    _patch_dataset_formatting(monkeypatch)
    trainer = _dataset_loader_self()
    snapshot = tmp_path / "datasets--org--dataset" / "snapshots" / "dataset-commit"
    snapshot.mkdir(parents = True)
    (snapshot / "train.parquet").write_bytes(b"dataset")
    stream = SimpleNamespace(
        info = SimpleNamespace(
            download_checksums = {
                "hf://datasets/org/dataset@dataset-commit/train.parquet": {
                    "num_bytes": 7,
                    "checksum": None,
                }
            }
        ),
        take = lambda _count: [{"text": "example"}],
    )

    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda **kw: [tmp_path])
    monkeypatch.setattr(
        "core.training.trainer.load_dataset",
        lambda **kwargs: (
            stream if kwargs.get("streaming") else pytest.fail("eager download should not run")
        ),
    )

    result = trainer.load_and_format_dataset(
        "org/dataset",
        dataset_slice_end = 0,
    )

    assert result is not None
    assert trainer.dataset_snapshot_path == str(snapshot.resolve())
    assert trainer.dataset_loaded_from_exact_snapshot is True


@pytest.mark.parametrize(
    "cached_eval_error",
    [
        FileNotFoundError("validation"),
        ValueError("Unknown split \"validation\". Should be one of ['train']."),
    ],
)
def test_cached_explicit_eval_failure_reloads_remote_pair(monkeypatch, cached_eval_error):
    from hub.utils import dataset_cache

    _patch_dataset_formatting(monkeypatch)
    trainer = _dataset_loader_self()
    cache_calls: list[str] = []
    remote_calls: list[str] = []
    cached_train = _SizedDataset(40, ("train", "validation"))
    remote_train = _SizedDataset(50, ("train", "validation"))
    remote_validation = _SizedDataset(20, ("train", "validation"))

    def load_cached(
        repo_id,
        local_path,
        *,
        subset,
        split,
        token = None,
    ):
        cache_calls.append(split)
        if split == "validation":
            raise cached_eval_error
        return cached_train

    def load_remote(*, path, split, **kwargs):
        remote_calls.append(split)
        return remote_validation if split == "validation" else remote_train

    monkeypatch.setattr(dataset_cache, "load_cached_hf_dataset", load_cached)
    monkeypatch.setattr("core.training.trainer.load_dataset", load_remote)

    result = trainer.load_and_format_dataset(
        "org/dataset",
        eval_split = "validation",
        eval_steps = 1,
        dataset_local_files_only = True,
        dataset_local_path = "/cache/snapshot",
    )

    assert result is not None
    assert result[0]["dataset"] is remote_train
    assert result[1] is remote_validation
    assert cache_calls == ["train", "validation"]
    assert remote_calls == ["train", "validation"]


class TestPreflightFirstBatch(unittest.TestCase):
    def test_float_input_ids_with_empty_template_suggests_instruct(self):
        ds = [{"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]}]
        inner = _FakeInnerTrainer(
            batch = {"input_ids": torch.zeros((1, 0), dtype = torch.float32)},
            train_dataset = ds,
        )
        s = _fake_self(
            inner = inner, model_name = "Qwen/Qwen2-VL-7B", tokenizer = _EmptyTemplateTokenizer()
        )
        msg = s._preflight_first_batch()
        self.assertIsNotNone(msg)
        self.assertIn("chat template", msg)
        self.assertIn("Qwen/Qwen2-VL-7B-Instruct", msg)
        self.assertIn("base (pretrained) model", msg)

    def test_no_instruct_hint_when_model_already_instruct(self):
        ds = [{"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]}]
        inner = _FakeInnerTrainer(
            batch = {"input_ids": torch.zeros((1, 0), dtype = torch.float32)},
            train_dataset = ds,
        )
        s = _fake_self(
            inner = inner, model_name = "org/Foo-Instruct", tokenizer = _EmptyTemplateTokenizer()
        )
        msg = s._preflight_first_batch()
        self.assertIsNotNone(msg)
        self.assertNotIn("such as", msg)  # no Instruct suggestion for an Instruct model
        self.assertIn("instruction-tuned variant", msg)

    def test_empty_int_input_ids_generic_message(self):
        inner = _FakeInnerTrainer(
            batch = {"input_ids": torch.zeros((1, 0), dtype = torch.long)},
            train_dataset = [{"text": "already tokenized path"}],
        )
        s = _fake_self(inner = inner, tokenizer = _RealTemplateTokenizer())
        msg = s._preflight_first_batch()
        self.assertIsNotNone(msg)
        self.assertIn("invalid token IDs", msg)
        self.assertNotIn("chat template", msg)

    def test_valid_batch_returns_none(self):
        inner = _FakeInnerTrainer(
            batch = {"input_ids": torch.randint(0, 1000, (2, 34), dtype = torch.long)},
        )
        s = _fake_self(inner = inner)
        self.assertIsNone(s._preflight_first_batch())

    def test_dataloader_error_is_surfaced(self):
        inner = _FakeInnerTrainer(dataloader_error = RuntimeError("boom"))
        s = _fake_self(inner = inner, model_name = "org/M")
        msg = s._preflight_first_batch()
        self.assertIsNotNone(msg)
        self.assertIn("failed to build the first training batch", msg)
        self.assertIn("org/M", msg)

    def test_missing_input_ids_does_not_false_positive(self):
        inner = _FakeInnerTrainer(batch = {"pixel_values": torch.zeros((1, 3))})
        s = _fake_self(inner = inner)
        self.assertIsNone(s._preflight_first_batch())


class TestChatTemplateRendersEmpty(unittest.TestCase):
    def _self(self, *, train_dataset, tokenizer):
        inner = _FakeInnerTrainer(train_dataset = train_dataset)
        return _fake_self(inner = inner, tokenizer = tokenizer)

    def test_empty_render_detected(self):
        ds = [{"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]}]
        s = self._self(train_dataset = ds, tokenizer = _EmptyTemplateTokenizer())
        self.assertTrue(s._chat_template_renders_empty())

    def test_nonempty_render_not_flagged(self):
        ds = [{"messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]}]
        s = self._self(train_dataset = ds, tokenizer = _RealTemplateTokenizer())
        self.assertFalse(s._chat_template_renders_empty())

    def test_no_messages_key_not_flagged(self):
        s = self._self(train_dataset = [{"text": "raw"}], tokenizer = _EmptyTemplateTokenizer())
        self.assertFalse(s._chat_template_renders_empty())


def _clear_trainer_module(package: str):
    sys.modules.pop(f"{package}.trainer", None)
    pkg = sys.modules.get(package)
    if pkg is not None and hasattr(pkg, "trainer"):
        delattr(pkg, "trainer")


def _set_training_platform(monkeypatch, package: str, backend: str):
    training_mod = importlib.import_module(f"{package}.training")
    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "DEVICE", None)
    monkeypatch.setattr(
        training_mod.platform,
        "system",
        lambda: "Darwin" if backend == "mlx" else "Linux",
    )
    monkeypatch.setattr(
        training_mod.platform,
        "machine",
        lambda: "arm64" if backend == "mlx" else "x86_64",
    )


def _load_trainer_module(
    monkeypatch,
    backend: str,
    package: str = "core.training",
):
    _set_training_platform(monkeypatch, package, backend)
    _clear_trainer_module(package)
    with _stubbed():
        if package in sys.modules:
            importlib.reload(sys.modules[package])
        trainer_mod = importlib.import_module(f"{package}.trainer")
        training_mod = importlib.import_module(f"{package}.training")
    monkeypatch.setattr(
        training_mod._MLXTrainerAdapter,
        "_activate_transformers_for_model",
        lambda self, model_name, hf_token: None,
    )
    return trainer_mod


class _ExitedProc:
    def join(self, timeout = None):
        return None

    def is_alive(self):
        return False


class _TerminableProc:
    def __init__(self):
        self.terminated = False
        self._done = threading.Event()

    def join(self, timeout = None):
        self._done.wait(timeout = timeout or 5)

    def is_alive(self):
        return not self.terminated

    def terminate(self):
        self.terminated = True
        self._done.set()


def test_unsloth_trainer_dispatches_for_mlx_and_torch(monkeypatch):
    trainer_mod = _load_trainer_module(monkeypatch, "mlx")

    mlx_trainer = trainer_mod.UnslothTrainer()

    assert type(mlx_trainer).__module__ == "core.training.training"
    assert mlx_trainer.get_training_progress().status_message == "Ready to train"

    trainer_mod = _load_trainer_module(monkeypatch, "torch")

    assert trainer_mod.UnslothTrainer().__class__ is trainer_mod.UnslothTrainer


def test_cli_mlx_trainer_activates_before_importing_trainer():
    repo_root = Path(__file__).resolve().parents[3]
    script = """
import json
import sys
import unsloth_cli.commands.train as train_cmd
from studio.backend.core.training import training as training_mod
from utils.hardware import hardware as hw

training_mod.platform.system = lambda: "Darwin"
training_mod.platform.machine = lambda: "arm64"
hw.DEVICE = None
events = []

def fake_activate(model_name, hf_token):
    events.append({
        "model_name": model_name,
        "trainer_loaded": "studio.backend.core.training.trainer" in sys.modules,
    })

train_cmd._activate_mlx_transformers = fake_activate
trainer = train_cmd._create_cli_trainer("mlx-community/Qwen3-0.6B-4bit", None)
print(json.dumps({
    "trainer_module": type(trainer).__module__,
    "events": events,
}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), str(repo_root / "studio" / "backend"), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd = repo_root,
        env = env,
        text = True,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        check = True,
    )
    payload = json.loads(result.stdout)

    assert payload["trainer_module"] == "studio.backend.core.training.training"
    assert payload["events"] == [
        {"model_name": "mlx-community/Qwen3-0.6B-4bit", "trainer_loaded": False}
    ]


def test_mlx_adapter_builds_config_and_reports_completion(tmp_path, monkeypatch):
    trainer_mod = _load_trainer_module(monkeypatch, "mlx")
    captured = {}

    def fake_run_worker(config, event_queue, stop_queue):
        captured["config"] = config
        event_queue.put({"type": "progress", "step": 1, "total_steps": 1, "loss": 0.25})
        event_queue.put(
            {"type": "complete", "status_message": "done", "output_dir": config["output_dir"]}
        )

    trainer = trainer_mod.UnslothTrainer()
    monkeypatch.setattr(trainer, "_run_mlx_worker", fake_run_worker)

    assert trainer.load_model("mlx-community/Qwen3-0.6B-4bit", max_seq_length = 1024)
    assert trainer.prepare_model_for_training(use_lora = False)
    dataset, eval_dataset = trainer.load_and_format_dataset("org/dataset")
    output_dir = tmp_path / "mlx-out"

    assert trainer.start_training(
        dataset = dataset,
        eval_dataset = eval_dataset,
        output_dir = output_dir,
        project_name = "Sales Assistant",
        max_steps = 1,
        learning_rate = 3e-4,
    )
    trainer.training_thread.join(timeout = 5)

    progress = trainer.get_training_progress()
    config = captured["config"]
    assert progress.is_completed
    assert progress.output_dir == str(output_dir.resolve())
    progress.status_message = "mutated"
    assert trainer.get_training_progress().status_message == "done"
    assert config["model_name"] == "mlx-community/Qwen3-0.6B-4bit"
    assert config["project_name"] == "Sales Assistant"
    assert config["hf_dataset"] == "org/dataset"
    assert config["training_type"] == "Full Finetuning"
    assert config["load_in_4bit"] is False
    assert config["max_seq_length"] == 1024
    assert config["learning_rate"] == 3e-4
    assert config["output_dir"] == str(output_dir.resolve())
    assert config["allow_external_output_dir"] is True


def test_mlx_worker_helpers_cover_cli_paths(tmp_path, monkeypatch):
    _load_trainer_module(monkeypatch, "mlx")
    from core.training.worker import (
        _resolve_mlx_local_dataset_files,
        _resolve_mlx_output_dir,
    )

    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"text":"hello"}\n', encoding = "utf-8")
    monkeypatch.chdir(tmp_path)

    assert _resolve_mlx_local_dataset_files(["train.jsonl"]) == [str(dataset)]
    assert _resolve_mlx_output_dir(
        {"output_dir": "cli-out", "allow_external_output_dir": True},
        "mlx-community/Qwen3-0.6B-4bit",
    ) == str((tmp_path / "cli-out").resolve())


def test_run_mlx_training_process_applies_side_effects_before_hardware_detection(monkeypatch):
    _load_trainer_module(monkeypatch, "mlx")
    from core.training import worker
    from utils.hardware import hardware as hw

    order = []

    def fake_activate(model_name, hf_token):
        order.append(("activate", model_name, hf_token))

    def fake_validate(config, event_queue):
        order.append("validate")
        assert os.environ["HF_HUB_DISABLE_XET"] == "1"
        assert os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "0"
        return True

    def fake_detect_hardware():
        order.append("detect")
        hw.DEVICE = hw.DeviceType.CPU
        return hw.DEVICE

    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising = False)
    monkeypatch.delenv("HF_HUB_ENABLE_HF_TRANSFER", raising = False)
    monkeypatch.setattr(worker, "_validate_training_worker_config", fake_validate)
    monkeypatch.setattr(worker, "_activate_transformers_version_or_warn", fake_activate)
    monkeypatch.setattr(hw, "detect_hardware", fake_detect_hardware)

    event_queue = queue.Queue()
    worker.run_mlx_training_process(
        event_queue = event_queue,
        stop_queue = queue.Queue(),
        config = {"model_name": "mlx-community/Gemma-4-12B", "disable_xet": True},
    )

    event = event_queue.get_nowait()
    assert order == ["validate", ("activate", "mlx-community/Gemma-4-12B", None), "detect"]
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "0"
    assert "MLX training requires Apple Silicon" in event["error"]


def test_run_mlx_training_process_rejects_untrainable_format_before_side_effects(monkeypatch):
    _load_trainer_module(monkeypatch, "mlx")
    from core.training import worker
    from utils.hardware import hardware as hw

    monkeypatch.setattr(
        worker,
        "_activate_transformers_version_or_warn",
        lambda *_args: pytest.fail("activation must not run"),
    )
    monkeypatch.setattr(
        hw,
        "detect_hardware",
        lambda: pytest.fail("hardware detection must not run"),
    )
    event_queue = queue.Queue()

    worker.run_mlx_training_process(
        event_queue = event_queue,
        stop_queue = queue.Queue(),
        config = {"model_name": "org/model", "model_format": "gguf"},
    )

    assert "GGUF" in event_queue.get_nowait()["error"]


def test_run_mlx_training_process_rejects_invalid_exact_pin_before_side_effects(
    tmp_path, monkeypatch
):
    _load_trainer_module(monkeypatch, "mlx")
    from core.training import worker
    from utils.hardware import hardware as hw

    monkeypatch.setattr(
        worker,
        "_activate_transformers_version_or_warn",
        lambda *_args: pytest.fail("activation must not run"),
    )
    monkeypatch.setattr(
        hw,
        "detect_hardware",
        lambda: pytest.fail("hardware detection must not run"),
    )
    event_queue = queue.Queue()

    worker.run_mlx_training_process(
        event_queue = event_queue,
        stop_queue = queue.Queue(),
        config = {
            "model_name": "org/model",
            "model_snapshot_path": str(tmp_path / "missing"),
            "load_in_4bit": False,
            "require_exact_model_resource": True,
        },
    )

    assert "exact model snapshot" in event_queue.get_nowait()["error"]


def test_run_mlx_training_process_skips_duplicate_config_validation(monkeypatch):
    _load_trainer_module(monkeypatch, "mlx")
    from core.training import worker
    from utils.hardware import hardware as hw

    monkeypatch.setattr(
        worker,
        "_validate_training_worker_config",
        lambda *_args: pytest.fail("prevalidated config must not be checked twice"),
    )

    def fake_detect_hardware():
        hw.DEVICE = hw.DeviceType.CPU
        return hw.DEVICE

    monkeypatch.setattr(hw, "detect_hardware", fake_detect_hardware)
    event_queue = queue.Queue()

    worker.run_mlx_training_process(
        event_queue = event_queue,
        stop_queue = queue.Queue(),
        config = {"model_name": "org/model"},
        transformers_activated = True,
        config_prevalidated = True,
    )

    assert "MLX training requires Apple Silicon" in event_queue.get_nowait()["error"]


def test_mlx_worker_callsites_select_config_validation_policy(monkeypatch):
    import inspect

    _load_trainer_module(monkeypatch, "mlx")
    from core.training import training, worker

    outer_source = inspect.getsource(worker.run_training_process)
    adapter_source = inspect.getsource(training._MLXTrainerAdapter._run_mlx_worker)

    assert outer_source.count("_validate_training_worker_config(config, event_queue)") == 1
    assert "config_prevalidated = True" in outer_source
    assert "config_prevalidated" not in adapter_source


if __name__ == "__main__":
    unittest.main()


def test_a_cached_spark_snapshot_root_still_gets_the_llm_subfolder(tmp_path):
    """An offline or cache-pinned snapshot root has an LLM/ child, which the previous check
    read as "already at the tokenizer" and sent AutoTokenizer at the root, which has none."""
    from core.training.trainer import _spark_tts_tokenizer_kwargs

    snapshot = tmp_path / "snapshots" / "abc123"
    (snapshot / "LLM").mkdir(parents = True)

    assert _spark_tts_tokenizer_kwargs("bicodec", str(snapshot)) == {"subfolder": "LLM"}
    assert _spark_tts_tokenizer_kwargs("bicodec", "unsloth/Spark-TTS-0.5B") == {"subfolder": "LLM"}
    # Already pointed at the tokenizer directory.
    assert _spark_tts_tokenizer_kwargs("bicodec", str(snapshot / "LLM")) == {}
    # A local checkpoint holding its own tokenizer.
    local = tmp_path / "my-ft"
    local.mkdir()
    (local / "tokenizer_config.json").write_text("{}", encoding = "utf-8")
    assert _spark_tts_tokenizer_kwargs("bicodec", str(local)) == {}
    # Not Spark at all.
    assert _spark_tts_tokenizer_kwargs("snac", str(snapshot)) == {}

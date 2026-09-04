# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The worker-count bound applied to unsloth_zoo's train_on_responses_only.

The zoo sizes its own dataset.map() workers with the uncapped heuristic issue
#2693 is about, so unsloth.chat_templates wraps it. These tests pin the two
things that make the wrapper non-obvious: None means "auto" to the zoo, not
"in-process", and an explicit count switches off its small-split guard.
"""

import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_helper():
    spec = importlib.util.spec_from_file_location(
        "_unsloth_dnp", REPO_ROOT / "unsloth" / "dataset_num_proc.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dnp = _load_helper()


class _Split:
    """Minimal sized stand-in for a datasets.Dataset."""

    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return self._rows


class _Unsized:
    """Stands in for an IterableDataset, whose length the zoo cannot read."""

    def __len__(self):
        raise TypeError("unsized")


class _Trainer:
    def __init__(
        self,
        train_dataset = None,
        eval_dataset = None,
    ):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset


BIG = dnp.ZOO_MIN_ROWS_FOR_MULTIPROC
SMALL = dnp.ZOO_MIN_ROWS_FOR_MULTIPROC - 1


@pytest.fixture(autouse = True)
def _reset(monkeypatch):
    dnp.reset_warning_state()
    monkeypatch.delenv(dnp.NUM_PROC_ENV_VAR, raising = False)
    # Pin CPUs, memory and both start methods so the assertions are about the wrapper, not this machine.
    # Auto is min(max(cpus // 2, AUTO_NUM_PROC_CAP)), so reaching the cap needs >= 2 * AUTO_NUM_PROC_CAP usable CPUs
    pytest.importorskip("psutil")
    monkeypatch.setattr(dnp, "_usable_cpus", lambda: 64)
    monkeypatch.setattr(dnp, "_affordable_workers", lambda: 1000)
    monkeypatch.setattr(dnp, "multiprocessing_start_method", lambda: "fork")
    # The zoo's own auto path reads stdlib multiprocessing;
    # The disagreement is covered in test_dataset_num_proc.py.
    monkeypatch.setattr(dnp, "_zoo_auto_sizer_forks", lambda: True)
    yield
    dnp.reset_warning_state()


# ── The zoo's threshold, duplicated in ZOO_MIN_ROWS_FOR_MULTIPROC ──


def _zoo_source():
    """Read unsloth_zoo's dataset_utils source without importing it.

    Importing the package pulls torch, which is not always loadable here, and
    these two checks only need the text. find_spec on the submodule would import
    unsloth_zoo, so locate the top level package and read the file off disk.
    """
    spec = importlib.util.find_spec("unsloth_zoo")
    locations = list(getattr(spec, "submodule_search_locations", None) or [])
    if spec is None or not locations:
        pytest.skip("unsloth_zoo is not installed")
    source = Path(locations[0]) / "dataset_utils.py"
    if not source.is_file():
        pytest.skip("unsloth_zoo.dataset_utils not found on disk")
    return source.read_text()


def test_zoo_threshold_constant_has_not_drifted():
    """ZOO_MIN_ROWS_FOR_MULTIPROC mirrors a local inside the zoo function.

    It cannot be imported, so this canary is the only thing between a zoo change
    and silently removing its small-split guard.
    """
    match = re.search(r"_MIN_ROWS_FOR_MULTIPROC\s*=\s*([0-9_]+)", _zoo_source())
    assert match is not None, (
        "unsloth_zoo no longer defines _MIN_ROWS_FOR_MULTIPROC; "
        "resolve_responses_only_num_proc assumes it exists to decide when "
        "substituting a worker count is safe"
    )
    assert int(match.group(1).replace("_", "")) == dnp.ZOO_MIN_ROWS_FOR_MULTIPROC


def test_zoo_still_treats_none_as_auto_not_serial():
    """The reason None cannot be forwarded to the zoo as an in-process request."""
    assert "_num_proc_was_auto = num_proc is None or type(num_proc) is not int" in _zoo_source()


# ── Explicit counts ──


def test_explicit_count_is_bounded(monkeypatch):
    monkeypatch.setattr(dnp, "_affordable_workers", lambda: 4)
    assert dnp.resolve_responses_only_num_proc(_Trainer(_Split(BIG)), 64) == 4


def test_explicit_count_survives_when_memory_allows():
    assert dnp.resolve_responses_only_num_proc(_Trainer(_Split(BIG)), 6) == 6


def test_explicit_one_stays_one_and_never_becomes_none():
    """None would read as 'auto' to the zoo and inflate to the auto count."""
    assert dnp.resolve_responses_only_num_proc(_Trainer(_Split(BIG)), 1) == 1


def test_explicit_count_serialised_by_memory_becomes_one_not_none(monkeypatch):
    monkeypatch.setattr(dnp, "_affordable_workers", lambda: 0)
    assert dnp.resolve_responses_only_num_proc(_Trainer(_Split(BIG)), 32) == 1


def test_explicit_count_ignores_split_size():
    """The zoo already skips its guard for explicit counts, so we take nothing."""
    assert dnp.resolve_responses_only_num_proc(_Trainer(_Split(SMALL)), 6) == 6


# ── Auto counts: the zoo's small-split guard must survive ──


def test_auto_small_split_passes_none_through():
    assert dnp.resolve_responses_only_num_proc(_Trainer(_Split(SMALL)), None) is None


def test_auto_large_split_gets_the_bounded_count():
    resolved = dnp.resolve_responses_only_num_proc(_Trainer(_Split(BIG)), None)
    assert resolved == dnp.AUTO_NUM_PROC_CAP


def test_auto_uses_the_largest_split_not_the_smallest():
    """A big train split must still be bounded when eval is small."""
    trainer = _Trainer(train_dataset = _Split(BIG), eval_dataset = _Split(10))
    assert dnp.resolve_responses_only_num_proc(trainer, None) == dnp.AUTO_NUM_PROC_CAP


def test_auto_unsized_split_passes_none_through():
    trainer = _Trainer(train_dataset = _Unsized())
    assert dnp.resolve_responses_only_num_proc(trainer, None) is None


@pytest.mark.parametrize(
    "trainer",
    [
        _Trainer(train_dataset = _Split(BIG), eval_dataset = _Unsized()),
        _Trainer(train_dataset = _Unsized(), eval_dataset = _Split(BIG)),
        _Trainer(train_dataset = _Split(BIG), eval_dataset = {"a": _Unsized()}),
    ],
    ids = ["unsized-eval", "unsized-train", "unsized-eval-dict"],
)
def test_an_unsized_split_does_not_hide_a_large_sized_one(trainer):
    """Regression: one unsized split disabled the bound for every other split.

    An unsized split used to abandon the measurement and return None, which the
    zoo reads as "auto", not "in-process", so it sized the *sized* split with its
    own uncapped min(max(cpu_count + 4, 2), 64). The unsized one can never use
    workers anyway (the zoo's IterableDataset branch passes no num_proc).
    """
    assert dnp.resolve_responses_only_num_proc(trainer, None) == dnp.AUTO_NUM_PROC_CAP


def test_auto_dict_eval_dataset_is_unpacked():
    trainer = _Trainer(train_dataset = _Split(10), eval_dataset = {"a": _Split(BIG)})
    assert dnp.resolve_responses_only_num_proc(trainer, None) == dnp.AUTO_NUM_PROC_CAP


def test_no_trainer_passes_none_through():
    assert dnp.resolve_responses_only_num_proc(None, None) is None


def test_trainer_without_datasets_passes_none_through():
    assert dnp.resolve_responses_only_num_proc(_Trainer(), None) is None


def test_bool_is_treated_as_auto_like_the_zoo_does():
    """type(True) is not int by the zoo's test, so it counts as auto."""
    assert dnp.resolve_responses_only_num_proc(_Trainer(_Split(SMALL)), True) is True


def test_env_override_still_wins_for_explicit_values(monkeypatch):
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, "16")
    monkeypatch.setattr(dnp, "_affordable_workers", lambda: 2)
    assert dnp.resolve_responses_only_num_proc(_Trainer(_Split(BIG)), 4) == 16


@pytest.mark.parametrize(
    "trainer",
    [
        _Trainer(train_dataset = _Split(SMALL)),
        _Trainer(train_dataset = _Unsized()),
        _Trainer(),
    ],
    ids = ["small-split", "unsized-split", "no-splits"],
)
def test_env_override_wins_on_the_split_size_shortcut(monkeypatch, trainer):
    """The escape hatch must win everywhere, including the early return.

    The shortcut for splits the zoo would not have parallelized used to return
    before UNSLOTH_DATASET_NUM_PROC was read, dropping a count set by a user who
    had just been told to set it.
    """
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, "3")
    assert dnp.resolve_responses_only_num_proc(trainer, None) == 3


@pytest.fixture
def _spawn(monkeypatch):
    # Both modules on spawn, the ordinary Windows host. The interesting case is when they
    # disagree, which is what _split below covers.
    monkeypatch.setattr(dnp, "multiprocessing_start_method", lambda: "spawn")
    monkeypatch.setattr(dnp, "_zoo_auto_sizer_forks", lambda: False)


@pytest.fixture
def _split(monkeypatch):
    # multiprocess on spawn, stdlib still on fork: a None here is "size it for me" to the zoo,
    # and datasets then builds that pool on the spawn context.
    monkeypatch.setattr(dnp, "multiprocessing_start_method", lambda: "spawn")
    monkeypatch.setattr(dnp, "_zoo_auto_sizer_forks", lambda: True)


@pytest.mark.parametrize(
    "requested",
    [None, 32, 1],
    ids = ["auto", "explicit-count", "explicit-one"],
)
def test_serial_is_none_not_one_on_spawn(_spawn, requested):
    """On spawn the zoo must be left to veto, not handed a Pool(1).

    ``1`` is not in-process on ``datasets`` >= 4.1, and under spawn each of those
    children re-imports the user's ``__main__`` (#3211 / #3397). ``None`` is safe
    precisely because it is *not* serial to the zoo: its auto path runs its own
    non-fork veto and lands in-process. That holds only while the zoo's check
    agrees, which is why _spawn pins both modules.
    """
    assert dnp.resolve_responses_only_num_proc(_Trainer(_Split(BIG)), requested) is None


@pytest.mark.parametrize(
    "requested",
    [None, 32, 1],
    ids = ["auto", "explicit-count", "explicit-one"],
)
def test_one_worker_when_only_multiprocess_is_on_spawn(_split, requested):
    """The zoo's veto reads stdlib multiprocessing, so it would not fire here.

    A None would be auto-sized to cpu_count + 4 and datasets would build that
    pool on the spawn context. One worker is the smallest request the zoo honours
    verbatim: still a Pool(1), but not dozens of them.
    """
    assert dnp.resolve_responses_only_num_proc(_Trainer(_Split(BIG)), requested) == 1


def test_env_forced_serial_on_a_large_split_is_one_on_fork(monkeypatch):
    """The fork side of the same branch, where 1 is the best available value.

    The zoo's row guard cannot help on a large split and None would be read as
    "auto" and inflated, so one forked worker is the floor this wrapper can
    express. Pinned so the spawn tests above cannot be "fixed" by making every
    start method return None.
    """
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, "0")
    assert dnp.resolve_responses_only_num_proc(_Trainer(_Split(BIG)), None) == 1


def test_env_explicit_count_is_not_downgraded_on_spawn(_spawn, monkeypatch):
    """The escape hatch is deliberate, so it keeps bypassing the veto."""
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, "3")
    assert dnp.resolve_responses_only_num_proc(_Trainer(_Split(BIG)), None) == 3


@pytest.mark.parametrize("raw", ["0", "none", "false"])
def test_env_forced_in_process_leaves_the_zoo_guard_in_charge(monkeypatch, raw):
    """UNSLOTH_DATASET_NUM_PROC=0 must not turn into a Pool(1) on a small split.

    The zoo's guard already yields None under its threshold, and None -- not the
    1 this function can express -- is the only value datasets runs in-process on
    every release, so honour the hatch by leaving the value alone.
    """
    monkeypatch.setenv(dnp.NUM_PROC_ENV_VAR, raw)
    assert dnp.resolve_responses_only_num_proc(_Trainer(_Split(SMALL)), None) is None

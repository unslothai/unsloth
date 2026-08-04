# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""The worker-count bound applied to unsloth_zoo's train_on_responses_only.

The zoo sizes its own dataset.map() workers with the uncapped heuristic that
issue #2693 is about, so unsloth.chat_templates wraps it. These tests pin the
two things that make the wrapper non-obvious: None means "auto" to the zoo (not
"in-process"), and an explicit count switches off the zoo's small-split guard.
"""

import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_helper():
    spec = importlib.util.spec_from_file_location(
        "_unsloth_dnp", REPO_ROOT / "unsloth" / "utils" / "dataset_num_proc.py"
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
    # Pin memory and start method so the assertions are about the wrapper's
    # policy, not about whatever this machine happens to have free.
    monkeypatch.setattr(dnp, "_affordable_workers", lambda: 1000)
    monkeypatch.setattr(dnp, "multiprocessing_start_method", lambda: "fork")
    yield
    dnp.reset_warning_state()


# ── The zoo's threshold, duplicated in ZOO_MIN_ROWS_FOR_MULTIPROC ──


def _zoo_source():
    """Read unsloth_zoo's dataset_utils source without importing it.

    Importing the package pulls torch, which is not always loadable in a test
    environment, and these two checks only need the text.
    """
    # find_spec on the submodule would import the unsloth_zoo package, which
    # patches torch on import and fails on some environments. Locate the top
    # level package (which is not executed) and read the file off disk.
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

    It cannot be imported, so if the zoo ever changes it this canary is the only
    thing standing between us and silently removing its small-split guard.
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


def test_auto_unsized_eval_split_is_conservative():
    trainer = _Trainer(train_dataset = _Split(BIG), eval_dataset = _Unsized())
    assert dnp.resolve_responses_only_num_proc(trainer, None) is None


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

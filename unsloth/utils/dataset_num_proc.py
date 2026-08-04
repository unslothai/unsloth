# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single source of truth for the ``dataset_num_proc`` used by ``Dataset.map()``.

This used to be four copies of the same heuristic pasted into generated trainer
source (``unsloth/models/rl.py``) and into ``unsloth_zoo.dataset_utils``. They
drifted, and two of the copies were wrong in ways that produced the crash in
https://github.com/unslothai/unsloth/issues/2693:

1. They asked ``multiprocessing.get_start_method()`` whether forking was
   available. ``datasets`` does not use ``multiprocessing`` -- ``Dataset.map``
   imports ``Pool`` from ``multiprocess`` (the dill fork), which keeps its own
   independent default context. Setting the stdlib start method therefore told
   the guard one thing while ``datasets`` did another.

2. They used ``1`` as the "disable multiprocessing" sentinel. That is only true
   on older ``datasets``. On ``datasets`` 4.3.0 (the Studio pin) ``map()`` takes
   the pool branch for any ``num_proc >= 1``, so ``num_proc = 1`` still forks a
   ``Pool(1)``. Only ``None`` is in-process on every supported version.
"""

from __future__ import annotations

import contextlib
import os
import sys
from typing import Iterator, Optional

__all__ = [
    "AUTO_NUM_PROC_CAP",
    "MEMORY_BUDGET_FRACTION",
    "NUM_PROC_ENV_VAR",
    "WORKER_MEMORY_BUDGET_GB",
    "ZOO_MIN_ROWS_FOR_MULTIPROC",
    "get_dataset_num_proc",
    "map_failure_diagnostics",
    "multiprocessing_start_method",
    "reset_warning_state",
    "resolve_responses_only_num_proc",
]

# Environment escape hatch. Set to a positive integer to force that worker count
# verbatim (no capping, no start-method veto), or to "0"/"none" to force
# in-process tokenization. Deliberate power users are never silently overridden.
NUM_PROC_ENV_VAR = "UNSLOTH_DATASET_NUM_PROC"

# Upper bound for the *auto* worker count. The previous heuristic allowed
# min(max(cpu_count + 4, 2), 64) -- up to 64 forked workers, each of which is
# handed its own dill-pickled copy of the tokenizer closure and an Arrow shard
# over a pipe. Tokenization stops scaling long before that and the memory and
# fork pressure is what users actually hit. Raise it via NUM_PROC_ENV_VAR.
#
# Measured on an 8000-row Arrow dataset with a Qwen2.5 fast tokenizer, 3 reps:
# num_proc None 6.3s, 1 9.5s, 8 8.1s, 32 14.2s, 64 21.7s. More workers were
# slower than none at every size tried, because each task ships a 5.4 MB dill
# pickle of the closure down a pipe.
AUTO_NUM_PROC_CAP = 8

# Peak RSS of a single tokenization worker, measured on the run above: ~680 MB,
# and flat across worker counts (each child gets its own full copy). Budget 1 GB
# apiece so a larger tokenizer or shard does not immediately invalidate this.
WORKER_MEMORY_BUDGET_GB = 1.0

# Share of *available* RAM that tokenization workers may spend. Tokenization is
# a preparation step, not the job, so leave most of the machine to the run that
# follows. This is what actually bounds the failure in issue #2693: workers get
# SIGKILLed by the OOM killer, and datasets reports only "One of the subprocesses
# has abruptly died during map operation" with the real cause discarded
# (datasets/utils/py_utils.py compares pool PIDs and never reads exit status).
MEMORY_BUDGET_FRACTION = 0.5

# Mirrors _MIN_ROWS_FOR_MULTIPROC in unsloth_zoo.dataset_utils.
# train_on_responses_only. Below this it runs a split in-process, because the
# workers cost more than they save -- but only when it picked the worker count
# itself; an explicit count skips that guard. resolve_responses_only_num_proc
# has to know the threshold to avoid taking the guard away. It cannot be
# imported: it is a local inside the Zoo function, so the duplication is
# unavoidable and a canary in tests/utils/test_dataset_num_proc.py fails if the
# Zoo ever moves it.
ZOO_MIN_ROWS_FOR_MULTIPROC = 5_000

# Warn at most once per process per distinct reason, so a script that builds
# several configs does not print the same line repeatedly.
_WARNED: set = set()


def reset_warning_state() -> None:
    """Clear the once-per-process warning memo. Intended for tests."""
    _WARNED.clear()


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(f"Unsloth: {message}")


def _unpinned_default_start_method(module) -> Optional[str]:
    """The default start method of a module that has not pinned one yet.

    ``get_all_start_methods()`` documents its first element as the default, and
    in the standard library it is. ``multiprocess`` copies that function
    verbatim -- including the darwin branch that lists ``spawn`` first -- but
    does **not** copy the darwin default that goes with it. Compare line 326 of
    each ``context.py`` on macOS::

        multiprocessing:  _default_context = DefaultContext(_concrete_contexts['spawn'])
        multiprocess:     _default_context = DefaultContext(_concrete_contexts['fork'])  #FIXME: spawn

    So on macOS ``multiprocess.get_all_start_methods()[0]`` says ``spawn`` while
    ``multiprocess.Pool`` actually forks, and since ``datasets`` builds its pool
    from ``multiprocess`` the list order is simply the wrong source. Read the
    default context's own name instead. Unlike ``get_context()`` that does not
    assign ``_actual_context``, so it keeps the no-side-effects property the
    caller depends on. Private attributes, hence the fallback.
    """
    try:
        methods = module.get_all_start_methods()
    except Exception:
        methods = []

    try:
        name = module.context._default_context._default_context._name
        # Only if it is actually available here. These are private attributes and
        # they are not consistent across builds: on a Windows runner this chain
        # answered 'fork' while get_all_start_methods() was ['spawn'], and a
        # start method the platform does not offer cannot be the one in use.
        # Believing it would have read Windows as forkable and let workers
        # through, which is the spawn re-import loop of #3211 / #3397.
        if isinstance(name, str) and name and (not methods or name in methods):
            return name
    except Exception:
        pass
    return methods[0] if methods else None


def multiprocessing_start_method() -> Optional[str]:
    """Return the start method ``datasets`` will actually use, or None.

    ``datasets.arrow_dataset`` does ``from multiprocess import Pool``, so
    ``multiprocess`` -- not stdlib ``multiprocessing`` -- is what decides how
    ``Dataset.map(num_proc = ...)`` spawns workers. Fall back to the stdlib
    module only when ``multiprocess`` is absent, and to None when neither can
    answer (in which case the caller treats forking as unavailable).

    Observing must not mutate: ``get_start_method(allow_none = False)`` pins
    ``_actual_context`` as a side effect, which would make a later
    ``set_start_method()`` raise. So ask with ``allow_none = True`` and, when
    nothing has been pinned yet, read the platform default off the module's own
    default context (see ``_unpinned_default_start_method``).
    """
    for module_name in ("multiprocess", "multiprocessing"):
        try:
            module = __import__(module_name)
            method = module.get_start_method(allow_none = True)
            if method is None:
                method = _unpinned_default_start_method(module)
            return method
        except Exception:
            continue
    return None


def _workers_unusable_reason() -> Optional[str]:
    """Why ``datasets`` workers cannot be used here, or None when they can.

    Two independent refusals, kept apart because they have different causes and
    only one of them is a property of the start method:

    * **Not ``fork``.** The child receives the tokenizer closure through a dill
      pickle and must re-import the dynamically generated trainer module, which
      is not importable by name, so a worker cannot come up at all.

    * **macOS, whatever the start method says.** CPython moved the macOS default
      from ``fork`` to ``spawn`` in 3.8 (bpo-33725) because "the fork start
      method should be considered unsafe as it can lead to crashes of the
      subprocess as macOS system libraries may start threads". ``multiprocess``
      never copied that change -- see ``_unpinned_default_start_method`` -- so
      ``datasets`` genuinely does fork on macOS, out of a parent that has
      already loaded Torch and its threaded BLAS. Reporting that truthfully is
      right; acting on it is not, so macOS stays in-process as a stated policy
      rather than as a side effect of a probe that happened to read ``spawn``.
    """
    start_method = multiprocessing_start_method()
    if start_method != "fork":
        return f"this process uses the {start_method!r} start method"
    if sys.platform == "darwin":
        return "'multiprocess' forks on macOS, which CPython documents as unsafe there (bpo-33725)"
    return None


def _affordable_workers() -> Optional[int]:
    """How many workers free RAM can cover, or None when it cannot be read."""
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
    except Exception:
        return None
    budget_gb = available_gb * MEMORY_BUDGET_FRACTION
    return int(budget_gb / WORKER_MEMORY_BUDGET_GB)


def _clamp_by_memory(num_proc: int) -> Optional[int]:
    """Bound a worker count by RAM. None means "do not use workers at all".

    This applies to explicitly requested counts as well as auto-sized ones. The
    old heuristic capped only the auto path, so a caller that passed a number --
    Studio passes ``max(1, cpu_count // 4)`` -- could ask for dozens of workers
    on a machine with no room for them. That is the shape that OOMs.
    """
    affordable = _affordable_workers()
    if affordable is None:
        # No psutil, so no memory reading. Honour the request rather than
        # silently serialising on a machine that may be perfectly capable.
        return num_proc

    if affordable < 2:
        _warn_once(
            "memory_serial",
            "not enough free memory for dataset tokenization workers "
            f"(~{WORKER_MEMORY_BUDGET_GB:g}GB each); tokenizing in-process.",
        )
        return None

    if affordable < num_proc:
        _warn_once(
            "memory_clamp",
            f"reducing dataset_num_proc {num_proc} -> {affordable} to fit free "
            f"memory (~{WORKER_MEMORY_BUDGET_GB:g}GB per worker). Set "
            f"{NUM_PROC_ENV_VAR} to override.",
        )
        return affordable

    return num_proc


def _auto_num_proc() -> Optional[int]:
    """Worker count to use when the caller did not ask for a specific one."""
    try:
        import psutil
        cpu_count = psutil.cpu_count() or 1
    except Exception:
        # No psutil means no CPU reading; stay conservative rather than guessing.
        return None

    return _clamp_by_memory(min(max(cpu_count // 2, 2), AUTO_NUM_PROC_CAP))


def _serial(serial_as_none: bool) -> Optional[int]:
    """The value meaning "run in-process", encoded for the calling layer.

    At a ``map()`` call site that is ``None``, the only value ``datasets`` runs
    in-process on every supported release. At the *config* layer it is ``1``,
    because a config ``None`` is read downstream as "auto-size me" -- writing
    ``None`` there would inflate a serial request back up to the auto worker
    count. The call site then turns that ``1`` into ``None``.

    That reasoning only holds while workers are usable at all, so the config
    sentinel is conditioned on ``_workers_unusable_reason``. Where they are not,
    nothing can inflate a config ``None``: every auto-sizer that reads it vetoes
    as well (this module at step 2 below, and the three copies in
    ``unsloth_zoo.dataset_utils``, which read stdlib ``multiprocessing`` and so
    see ``spawn`` on both Windows and macOS). Meanwhile ``1`` is unsafe there,
    because only the SFT
    map site is rewritten by ``rl_replacements.py`` -- TRL's DPO, KTO, CPO,
    ORPO, Reward and PRM trainers hand ``args.dataset_num_proc`` straight to
    ``Dataset.map``, where ``datasets`` >= 4.1 turns ``1`` into a ``Pool(1)``
    and each spawned child re-imports the user's ``__main__`` (the Windows
    spawn loops in unsloth #3211 / #3397). ``None`` is what those configs
    carried before this module existed, and it is the only value that builds no
    pool at all.
    """
    if serial_as_none:
        return None
    return None if _workers_unusable_reason() is not None else 1


def _from_environment() -> "tuple[bool, Optional[int]]":
    """Read NUM_PROC_ENV_VAR. Returns (was_set_and_valid, value)."""
    raw = os.environ.get(NUM_PROC_ENV_VAR)
    if raw is None:
        return False, None

    text = raw.strip()
    if text.lower() in ("", "0", "none", "null", "false"):
        return True, None

    try:
        value = int(text)
    except ValueError:
        _warn_once(
            "env_invalid",
            f"{NUM_PROC_ENV_VAR}={raw!r} is not an integer, ignoring it.",
        )
        return False, None

    if value < 0:
        _warn_once(
            "env_negative",
            f"{NUM_PROC_ENV_VAR}={raw!r} is negative, ignoring it.",
        )
        return False, None
    if value <= 1:
        return True, None
    return True, value


def get_dataset_num_proc(
    desired: Optional[int] = None, *, serial_as_none: bool = True
) -> Optional[int]:
    """Return a safe ``num_proc`` for ``Dataset.map()`` / ``Dataset.filter()``.

    ``None`` means "run in-process". It is the only value that avoids building a
    worker pool on every supported ``datasets`` release -- ``1`` does not, since
    ``datasets`` 4.x takes the pool branch for any ``num_proc >= 1``.

    Whatever is returned is bounded by free memory, including a count the caller
    asked for explicitly. Only ``NUM_PROC_ENV_VAR`` is exempt.

    Args:
        desired: The worker count the caller asked for, or None to auto-size.
        serial_as_none: How to encode "run in-process". True at a ``map()``
            call site, which yields ``None``. **False when writing back to a
            config**, which yields ``1``, because downstream readers
            (``unsloth_zoo.dataset_utils.sft_prepare_dataset``) treat a config
            ``None`` as "auto-size me", so storing ``None`` for a user who asked
            for ``1`` would silently inflate it to the auto worker count. The
            config keeps the user's intent; the call site makes it safe.

    Returns:
        A worker count >= 2, or the in-process sentinel for this layer (``None``
        at a call site, ``1`` at the config layer).
    """
    # 1. Explicit environment override wins over everything, uncapped and
    #    without the start-method veto, so a user who knows their workload is
    #    fork-safe is never silently downgraded.
    env_set, env_value = _from_environment()
    if env_set:
        # env_value None means the user asked for in-process, so it has to be
        # encoded for this layer like every other serial path. Returning a bare
        # None here would write None into the *config*, which downstream reads
        # as "auto-size me" -- so UNSLOTH_DATASET_NUM_PROC=0, the escape hatch
        # the dead-worker message tells people to use, would have inflated the
        # worker count instead of removing it.
        return _serial(serial_as_none) if env_value is None else env_value

    # 2. `datasets` workers receive the tokenizer closure through a dill pickle
    #    over a pipe. Under spawn/forkserver the child must also re-import the
    #    dynamically generated trainer module, which is not importable by name,
    #    so multiprocessing is unusable regardless of what was requested. macOS
    #    is refused for a separate reason -- see `_workers_unusable_reason`.
    unusable = _workers_unusable_reason()
    if unusable is not None:
        if isinstance(desired, int) and not isinstance(desired, bool) and desired > 1:
            _warn_once(
                "start_method",
                f"dataset_num_proc = {desired} cannot be used because "
                f"{unusable}. Falling back to single-process tokenization. "
                f"Set {NUM_PROC_ENV_VAR} to override.",
            )
        return _serial(serial_as_none)

    # 3. Normalise the "no multiprocessing" requests. `1` in particular is a
    #    trap: callers pass it meaning "serial", and datasets >= 4.1 hands them
    #    a Pool(1) instead. Nothing to clamp here, so return straight away.
    if isinstance(desired, int) and not isinstance(desired, bool) and desired <= 1:
        return _serial(serial_as_none)

    # 4. Auto-size when no usable request was made, then bound by memory. A
    #    non-int (or a bool, which is an int subclass) is not a request.
    if desired is None or not isinstance(desired, int) or isinstance(desired, bool):
        num_proc = _auto_num_proc()
    else:
        num_proc = _clamp_by_memory(desired)

    if num_proc is None:
        return _serial(serial_as_none)
    return num_proc


# The message datasets raises when a pool worker dies. It compares pool PIDs and
# never reads the child's exit status, so an OOM kill, a segfault and a genuine
# exception all arrive as this one string with the real cause discarded. That is
# why issue #2693 looks random and untraceable.
_WORKER_DIED = "subprocesses has abruptly died"


@contextlib.contextmanager
def map_failure_diagnostics(num_proc: Optional[int]) -> "Iterator[None]":
    """Re-raise a dead-worker error from ``Dataset.map`` with usable context.

    Names the start method, the worker count, and roughly what those workers
    cost, so the reader can tell an out-of-memory kill from anything else --
    none of which survives the original message. The cause is chained, so the
    child's traceback is still there for anyone who wants it.
    """
    try:
        yield
    except RuntimeError as exception:
        if _WORKER_DIED not in str(exception):
            raise

        workers = num_proc if isinstance(num_proc, int) else 1
        estimated_gb = workers * WORKER_MEMORY_BUDGET_GB
        raise RuntimeError(
            f"Unsloth: tokenization failed because a dataset worker died.\n"
            f"  dataset_num_proc = {num_proc!r} "
            f"({workers} worker{'' if workers == 1 else 's'}, "
            f"start method {multiprocessing_start_method()!r})\n"
            f"  Each worker holds its own copy of the tokenizer and its dataset "
            f"shard, so this run needed roughly {estimated_gb:g}GB on top of the "
            f"model. The most common cause is the out-of-memory killer.\n"
            f"  Retry with {NUM_PROC_ENV_VAR}=0 for the fewest workers this path "
            f"can use, or {NUM_PROC_ENV_VAR}=<n> to choose a count. That is "
            f"in-process everywhere except train_on_responses_only on a split "
            f"over {ZOO_MIN_ROWS_FOR_MULTIPROC:,} rows, where it is one worker: "
            f"the Zoo reads a bare None there as 'size it for me'.\n"
            f"  Original error: {exception}"
        ) from exception


def _largest_split_rows(trainer) -> Optional[int]:
    """Rows in the biggest *sized* split ``train_on_responses_only`` will map over.

    Returns None only when no split has a readable length at all. An unsized
    split is skipped rather than abandoning the whole measurement: the Zoo picks
    a worker count per split, and an unsized one can never use workers whatever
    it is handed (its ``IterableDataset`` branch passes no ``num_proc``, and
    ``_effective_num_proc`` returns None when ``len()`` raises). Letting one hide
    a large sized sibling is what left that sibling on the Zoo's uncapped auto
    count. ``eval_dataset`` may be a dict of named splits, so unpack that too.
    """
    if trainer is None:
        return None

    largest = 0
    measured = False
    for attribute in ("train_dataset", "eval_dataset"):
        dataset = getattr(trainer, attribute, None)
        if dataset is None:
            continue
        splits = list(dataset.values()) if isinstance(dataset, dict) else [dataset]
        for split in splits:
            if split is None:
                continue
            try:
                rows = len(split)
            except Exception:
                # Unsized, or a length that raises. The Zoo cannot parallelize
                # this split either way, so skip it -- do not let it mask a
                # sized sibling that the Zoo *would* parallelize.
                continue
            measured = True
            largest = max(largest, rows)
    return largest if measured else None


def resolve_responses_only_num_proc(trainer, num_proc):
    """Bound the worker count ``train_on_responses_only`` hands to ``map()``.

    ``unsloth_zoo.dataset_utils.train_on_responses_only`` still computes its own
    count with the uncapped ``min(max(cpu_count + 4, 2), 64)`` heuristic this
    module replaces everywhere else, so on a large host it forks dozens of
    workers that each receive a dill-pickled tokenizer closure. That is the
    shape behind issue #2693, and this function is what keeps it bounded. The
    Zoo is a separate package, so the fix has to work from the outside.

    Two properties of the Zoo's API constrain what can be expressed here:

    * **Serial is the config-layer sentinel, not the call-site one.** The Zoo
      reads ``None`` as "size it for me", so on ``fork`` handing it the ``None``
      that means serial at a ``map()`` call site would *inflate* the count to the
      auto value instead of removing it; ``1`` is the closest expressible
      request there. It still builds a ``Pool(1)`` on ``datasets`` >= 4.1, which
      cannot be fixed without changing the Zoo, but one forked worker is not the
      out-of-memory failure this guards against. Under spawn/forkserver that
      trade flips: ``1`` is the *worse* value, because each ``Pool(1)`` child
      re-imports the user's ``__main__`` (the spawn loops in #3211 / #3397),
      while ``None`` is safe -- the Zoo's auto path vetoes on a non-fork start
      method of its own accord and ends up in-process. That is exactly the
      distinction ``get_dataset_num_proc(..., serial_as_none = False)`` encodes,
      so this function defers to it rather than mapping serial to ``1`` by hand.
    * **An explicit count disables the Zoo's small-split guard**, which it
      applies per split. A trainer with a large train split and a small eval
      split would see the eval map pick up workers it would not have had. That
      is a second or two on the small split against tens of GB on the large one,
      so it is worth it -- but only when a split is actually big enough for the
      Zoo to have parallelized in the first place, which is what the row check
      below establishes.
    """
    # Mirror the Zoo's own test exactly (bools are not ints by this rule, and it
    # treats them as auto), so "explicit" means the same thing on both sides.
    was_auto = num_proc is None or type(num_proc) is not int

    if not was_auto:
        # Explicit counts already bypass the Zoo's small-split guard, so
        # bounding one takes nothing away that the caller had.
        return get_dataset_num_proc(num_proc, serial_as_none = False)

    rows = _largest_split_rows(trainer)
    if rows is None or rows < ZOO_MIN_ROWS_FOR_MULTIPROC:
        # An explicit escape hatch has to win here too, or the count the user
        # asked for is dropped on the floor by a shortcut they cannot see.
        env_set, env_value = _from_environment()
        if env_set and env_value is not None:
            return env_value
        # Otherwise the Zoo would have gone in-process anyway (that is also what
        # UNSLOTH_DATASET_NUM_PROC=0 asks for, and its guard yields None, which
        # is more in-process than the 1 this function can express). Leave the
        # value untouched so its guard keeps deciding, rather than substituting
        # a count that would switch multiprocessing back on for a small split.
        return num_proc

    return get_dataset_num_proc(None, serial_as_none = False)

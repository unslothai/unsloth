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
from typing import Iterator, Optional

__all__ = [
    "AUTO_NUM_PROC_CAP",
    "MEMORY_BUDGET_FRACTION",
    "NUM_PROC_ENV_VAR",
    "WORKER_MEMORY_BUDGET_GB",
    "get_dataset_num_proc",
    "map_failure_diagnostics",
    "multiprocessing_start_method",
    "reset_warning_state",
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
    nothing has been pinned yet, read the platform default off
    ``get_all_start_methods()``, whose first element is documented to be it.
    """
    for module_name in ("multiprocess", "multiprocessing"):
        try:
            module = __import__(module_name)
            method = module.get_start_method(allow_none = True)
            if method is None:
                methods = module.get_all_start_methods()
                method = methods[0] if methods else None
            return method
        except Exception:
            continue
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
    """
    return None if serial_as_none else 1


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
        return env_value

    # 2. `datasets` workers receive the tokenizer closure through a dill pickle
    #    over a pipe. Under spawn/forkserver the child must also re-import the
    #    dynamically generated trainer module, which is not importable by name,
    #    so multiprocessing is unusable regardless of what was requested.
    start_method = multiprocessing_start_method()
    if start_method != "fork":
        if isinstance(desired, int) and not isinstance(desired, bool) and desired > 1:
            _warn_once(
                "start_method",
                f"dataset_num_proc = {desired} needs the 'fork' start method, "
                f"but this process uses {start_method!r}. Falling back to "
                f"single-process tokenization. Set {NUM_PROC_ENV_VAR} to "
                f"override.",
            )
        return _serial(serial_as_none)

    # 3. Normalise the "no multiprocessing" requests. `1` in particular is a
    #    trap: callers pass it meaning "serial", and datasets >= 4.0 hands them
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
            f"  Tokenize in-process with {NUM_PROC_ENV_VAR}=0, or pick a worker "
            f"count with {NUM_PROC_ENV_VAR}=<n>.\n"
            f"  Original error: {exception}"
        ) from exception

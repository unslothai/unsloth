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

"""Fallback copy of the ``dataset_num_proc`` policy for ``Dataset.map()``.

``unsloth_zoo.dataset_num_proc`` is the source of truth, since generated trainer
source must not import back into the package that generated it. Every caller
tries the zoo first, so this copy exists only so that upgrading unsloth alone
still fixes the bug; ``test_the_two_copies_have_not_drifted`` keeps the two in
step, and this file can go once the zoo floor guarantees the module.

It replaces four drifted copies of the heuristic (generated source in
``unsloth/models/rl.py``, plus ``unsloth_zoo.dataset_utils``), two of them wrong
in ways that produced https://github.com/unslothai/unsloth/issues/2693:

1. They asked ``multiprocessing.get_start_method()`` about forking, but
   ``Dataset.map`` imports ``Pool`` from ``multiprocess`` (the dill fork), which
   keeps its own default context. The two read different settings.

2. They used ``1`` as the "disable multiprocessing" sentinel, but ``datasets``
   >= 4.1 (Studio pins 4.3.0) pools for any ``num_proc >= 1``, so only ``None``
   is in-process on every supported version.
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

# Escape hatch: a positive integer forces that count verbatim (no cap, no
# start-method veto), "0"/"none" forces in-process tokenization.
NUM_PROC_ENV_VAR = "UNSLOTH_DATASET_NUM_PROC"

# Upper bound for the *auto* count only; raise it via NUM_PROC_ENV_VAR. The old
# min(max(cpu_count + 4, 2), 64) forked up to 64 workers, each handed its own
# dill-pickled tokenizer closure and an Arrow shard over a pipe. Measured on
# 8000 rows, Qwen2.5 fast tokenizer, 3 reps: None 6.3s, 1 9.5s, 8 8.1s,
# 32 14.2s, 64 21.7s -- more workers were slower than none at every size.
AUTO_NUM_PROC_CAP = 8

# ~680 MB peak RSS per worker on that run, flat across counts; 1 GB for headroom.
WORKER_MEMORY_BUDGET_GB = 1.0

# Share of *available* RAM tokenization may spend; the rest belongs to the run
# that follows. This is what bounds issue #2693, where OOM-killed workers
# surface only as "One of the subprocesses has abruptly died during map
# operation", the real cause discarded.
MEMORY_BUDGET_FRACTION = 0.5

# Mirrors _MIN_ROWS_FOR_MULTIPROC, a local inside
# dataset_utils.train_on_responses_only (hence not importable): below it that
# helper maps a split in-process, unless handed an explicit count.
# resolve_responses_only_num_proc needs the threshold to keep that guard; a
# canary in tests/test_dataset_num_proc.py catches drift.
ZOO_MIN_ROWS_FOR_MULTIPROC = 5_000

# Warn at most once per process per distinct reason.
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

    ``get_all_start_methods()[0]`` is the documented default, but ``multiprocess``
    copies that function verbatim -- darwin branch listing ``spawn`` first --
    without copying the darwin default that goes with it, keeping ``fork`` as its
    ``_default_context`` (``#FIXME: spawn`` in its ``context.py``). So its list
    says ``spawn`` while its ``Pool`` forks, and ``datasets`` builds from
    ``multiprocess``. Read the default context's own name instead; unlike
    ``get_context()`` it does not assign ``_actual_context``, keeping the probe
    side-effect free. Private attributes, hence the fallback.
    """
    try:
        methods = module.get_all_start_methods()
    except Exception:
        methods = []

    try:
        name = module.context._default_context._default_context._name
        # Only if the platform actually offers it: a Windows runner answered
        # 'fork' while get_all_start_methods() was ['spawn'], which would read
        # Windows as forkable -- the spawn re-import loop of #3211 / #3397.
        if isinstance(name, str) and name and (not methods or name in methods):
            return name
    except Exception:
        pass
    return methods[0] if methods else None


def _module_start_method(module_name: str) -> Optional[str]:
    """One module's start method. Raises if the module cannot be read at all.

    Observing must not mutate: ``get_start_method(allow_none = False)`` pins
    ``_actual_context``, which would make a later ``set_start_method()`` raise.
    """
    module = __import__(module_name)
    method = module.get_start_method(allow_none = True)
    if method is None:
        method = _unpinned_default_start_method(module)
    return method


def multiprocessing_start_method() -> Optional[str]:
    """Return the start method ``datasets`` will actually use, or None.

    ``datasets.arrow_dataset`` does ``from multiprocess import Pool``, so
    ``multiprocess`` -- not stdlib ``multiprocessing`` -- decides how
    ``Dataset.map(num_proc = ...)`` spawns workers. Falls back to the stdlib
    module, then to None (the caller then treats forking as unavailable).
    """
    for module_name in ("multiprocess", "multiprocessing"):
        try:
            return _module_start_method(module_name)
        except Exception:
            continue
    return None


def _workers_unusable_reason() -> Optional[str]:
    """Why ``datasets`` workers cannot be used here, or None when they can.

    * **Not ``fork``.** The child must re-import the dynamically generated
      trainer module, which has no importable name, so it cannot come up at all.

    * **macOS, whatever the start method says.** CPython moved the macOS default
      to ``spawn`` in 3.8 (bpo-33725) because forking there "can lead to crashes
      of the subprocess as macOS system libraries may start threads".
      ``multiprocess`` never copied that, so ``datasets`` genuinely forks on
      macOS out of a parent holding Torch and a threaded BLAS. Reporting that
      truthfully is right; acting on it is not, so macOS stays in-process as a
      stated policy rather than as a side effect of the probe.
    """
    start_method = multiprocessing_start_method()
    if start_method != "fork":
        return f"this process uses the {start_method!r} start method"
    if sys.platform == "darwin":
        return "'multiprocess' forks on macOS, which CPython documents as unsafe there (bpo-33725)"
    return None


def _cgroup_limits() -> "tuple[Optional[int], Optional[float]]":
    """This process's cgroup memory ceiling in bytes and CPU ceiling in cores.

    ``(None, None)`` off Linux or outside a container. Reuses the readers in
    ``hf_xet_tuning`` rather than parsing ``/sys/fs/cgroup`` again: they already
    handle v1 against v2, the ``/proc/self/cgroup`` path walk and the "unlimited"
    sentinels. Imported lazily, so this module still loads on its own.
    """
    try:
        from unsloth_zoo.hf_xet_tuning import cgroup_cpu_limit, cgroup_memory_limit
        return cgroup_memory_limit(), cgroup_cpu_limit()
    except Exception:
        return None, None


def _cgroup_memory_used() -> Optional[int]:
    """Bytes this cgroup is already using, or None when that cannot be read.

    Only the well-known roots: runc bind-mounts the container's own cgroup there,
    which is the case this exists for. Without it the whole limit would look free
    on a container that has already spent most of it.
    """
    for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            with open(path, "r") as handle:
                return int(handle.readline().strip())
        except Exception:
            continue
    return None


def _available_memory_gb() -> Optional[float]:
    """Free RAM this process may actually use, or None when it cannot be read.

    ``psutil.virtual_memory().available`` reports the HOST inside a container, so
    a 2GB pod on a 512GB box read as having room for the full worker set and got
    OOM-killed -- the exact failure the memory ceiling exists to prevent.
    """
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
    except Exception:
        return None

    limit, _ = _cgroup_limits()
    if limit is not None:
        used = _cgroup_memory_used()
        free = limit - used if used is not None else limit
        available_gb = min(available_gb, max(free, 0) / (1024**3))
    return available_gb


def _usable_cpus() -> Optional[int]:
    """CPUs this process may actually run on, or None when that cannot be read.

    ``cpu_count()`` is the host's, so under ``taskset``, Slurm pinning or a
    Kubernetes CPU quota a one-core job would auto-size workers that then contend
    for that one core, making tokenization slower than doing it in-process.
    """
    try:
        import psutil
        cpus = psutil.cpu_count()
    except Exception:
        cpus = os.cpu_count()
    if not cpus:
        return None

    try:
        cpus = min(cpus, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass  # not Linux, or the call is unavailable: the host count stands

    _, quota = _cgroup_limits()
    if quota is not None and quota > 0:
        # A fractional quota still binds: Kubernetes "cpu: 500m" is 0.5 cores.
        cpus = min(cpus, max(1, int(quota)))
    return cpus


def _affordable_workers() -> Optional[int]:
    """How many workers free RAM can cover, or None when it cannot be read."""
    available_gb = _available_memory_gb()
    if available_gb is None:
        return None
    budget_gb = available_gb * MEMORY_BUDGET_FRACTION
    return int(budget_gb / WORKER_MEMORY_BUDGET_GB)


def _clamp_by_memory(num_proc: int) -> Optional[int]:
    """Bound a worker count by RAM. None means "do not use workers at all".

    Applies to explicit counts too: the old heuristic capped only the auto path,
    so a caller passing a number (Studio passes ``max(1, cpu_count // 4)``) could
    ask for dozens of workers on a machine with no room. That is what OOMs.
    """
    affordable = _affordable_workers()
    if affordable is None:
        # No memory reading, so honour the request rather than serialising a
        # machine that may be perfectly capable.
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
        import psutil  # noqa: F401  the memory clamp below needs it to mean anything
    except Exception:
        # No psutil means no memory reading; stay conservative.
        return None

    cpus = _usable_cpus()
    if cpus is None:
        return None
    if cpus <= 1:
        # Workers would only contend for the single core they are pinned to.
        return None

    return _clamp_by_memory(min(max(cpus // 2, 2), AUTO_NUM_PROC_CAP))


def _serial(serial_as_none: bool) -> Optional[int]:
    """The value meaning "run in-process", encoded for the calling layer.

    At a ``map()`` call site that is ``None``, the only value ``datasets`` runs
    in-process on every supported release. At the *config* layer it is ``1``,
    because a config ``None`` is read downstream as "auto-size me" and would
    inflate a serial request; the call site turns that ``1`` back into ``None``.

    That only holds while workers are usable, hence the veto check. Where they
    are not, nothing can inflate a config ``None`` -- every auto-sizer that reads
    it vetoes too -- while ``1`` is unsafe, since only the SFT map site is
    rewritten by ``rl_replacements.py``: TRL's DPO, KTO, CPO, ORPO, Reward and
    PRM trainers hand ``args.dataset_num_proc`` straight to ``Dataset.map``,
    whose ``Pool(1)`` child re-imports the user's ``__main__`` (#3211 / #3397).
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

    ``None`` means "run in-process" -- the only value that builds no worker pool
    on every supported ``datasets`` release, since 4.x pools for any
    ``num_proc >= 1``. Everything returned is bounded by free memory, explicit
    counts included; only ``NUM_PROC_ENV_VAR`` is exempt.

    Args:
        desired: The worker count the caller asked for, or None to auto-size.
        serial_as_none: How to encode "run in-process". True at a ``map()`` call
            site, yielding ``None``. **False when writing back to a config**,
            yielding ``1``, because downstream readers
            (``dataset_utils.sft_prepare_dataset``) treat a config ``None`` as
            "auto-size me" and would inflate a user's ``1``.

    Returns:
        A worker count >= 2, or this layer's in-process sentinel (``None`` at a
        call site, ``1`` at the config layer).
    """
    # 1. The environment override wins over everything, uncapped and unvetoed,
    #    so a user who knows their workload is fork-safe is never downgraded.
    env_set, env_value = _from_environment()
    if env_set:
        # In-process still has to be encoded for this layer, or
        # UNSLOTH_DATASET_NUM_PROC=0 -- the hatch the dead-worker message
        # recommends -- would inflate a config instead of removing workers.
        return _serial(serial_as_none) if env_value is None else env_value

    # 2. Workers are unusable whatever was requested; see
    #    `_workers_unusable_reason` for the two refusals.
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

    # 3. Normalise "no multiprocessing" requests. `1` is the trap: callers pass
    #    it meaning "serial" and datasets >= 4.1 hands them a Pool(1).
    if isinstance(desired, int) and not isinstance(desired, bool) and desired <= 1:
        return _serial(serial_as_none)

    # 4. Auto-size when no usable request was made, then bound by memory. A
    #    non-int, or a bool (an int subclass), is not a request.
    if desired is None or not isinstance(desired, int) or isinstance(desired, bool):
        num_proc = _auto_num_proc()
    else:
        num_proc = _clamp_by_memory(desired)

    if num_proc is None:
        return _serial(serial_as_none)
    return num_proc


# The message datasets raises when a pool worker dies. It never reads the child's
# exit status, so an OOM kill, a segfault and a genuine exception all arrive as
# this one string -- which is why #2693 looks untraceable.
_WORKER_DIED = "subprocesses has abruptly died"


@contextlib.contextmanager
def map_failure_diagnostics(num_proc: Optional[int]) -> "Iterator[None]":
    """Re-raise a dead-worker error from ``Dataset.map`` with usable context.

    Names the start method, the worker count and roughly what they cost, none of
    which survives the original message. The cause is chained.
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
            f"a bare None there reads as 'size it for me'.\n"
            f"  Original error: {exception}"
        ) from exception


def _largest_split_rows(trainer) -> Optional[int]:
    """Rows in the biggest *sized* split ``train_on_responses_only`` will map over.

    None only when no split has a readable length at all. Unsized splits are
    skipped rather than abandoning the measurement, since sizing is per split and
    an unsized one can never use workers anyway; letting one hide a large sized
    sibling left that sibling on the uncapped auto count. ``eval_dataset`` may be
    a dict of named splits, so unpack that too.
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
                # Skip rather than let it mask a sized sibling.
                continue
            measured = True
            largest = max(largest, rows)
    return largest if measured else None


def _zoo_auto_sizer_forks() -> bool:
    """Whether ``train_on_responses_only`` would pick workers for a bare ``None``.

    Its auto path asks stdlib ``multiprocessing``, not ``multiprocess`` -- the
    split this whole module is about. Where the two disagree, ``None`` is not
    "serial" to it, it is "size it for me", and ``datasets`` then builds that pool
    on the non-fork ``multiprocess`` context.
    """
    try:
        return _module_start_method("multiprocessing") == "fork"
    except Exception:
        return False


def _serial_for_the_zoo(value: Optional[int]) -> Optional[int]:
    """Re-encode a serial ``None`` for a reader that decides on the other module.

    ``None`` only means "no workers" to ``train_on_responses_only`` while its own
    check refuses them too. When it would not, ``1`` is the smallest request it
    honours verbatim: still a ``Pool(1)`` on ``datasets`` >= 4.1, but one child
    instead of the ``cpu_count + 4`` its auto path would have chosen.
    """
    if value is not None or not _zoo_auto_sizer_forks():
        return value
    _warn_once(
        "start_method_split",
        "'multiprocess' and 'multiprocessing' disagree about the start method "
        f"({multiprocessing_start_method()!r} against 'fork'), so "
        "train_on_responses_only is being held to one worker rather than left to "
        f"size itself. Set {NUM_PROC_ENV_VAR} to override.",
    )
    return 1


def resolve_responses_only_num_proc(trainer, num_proc):
    """Bound the worker count ``train_on_responses_only`` hands to ``map()``.

    ``dataset_utils.train_on_responses_only`` still auto-sizes with the uncapped
    ``min(max(cpu_count + 4, 2), 64)`` heuristic this module replaces everywhere
    else -- the shape behind issue #2693. Until that copy is wired onto this
    module the bound is applied from outside, and two properties of its API
    constrain what can be expressed:

    * **Serial is the config-layer sentinel, not the call-site one.** That helper
      reads ``None`` as "size it for me", so on ``fork`` handing it ``None``
      would *inflate* the count rather than remove it; ``1`` is the closest
      expressible request, and one forked worker is not the OOM this guards
      against. Under spawn the trade flips -- each ``Pool(1)`` child re-imports
      the user's ``__main__`` (#3211 / #3397), while ``None`` is safe because its
      auto path vetoes non-fork itself. That is exactly what
      ``serial_as_none = False`` encodes, so defer to it -- then re-check with
      ``_serial_for_the_zoo``, since that veto reads the other module.
    * **An explicit count disables its per-split small-split guard.** A small
      eval split alongside a large train split then picks up workers it would not
      have had: worth it only when a split is big enough to have been
      parallelized at all, which is what the row check below establishes.
    """
    # Mirror that helper's own test, so "explicit" means the same on both sides
    # (it treats bools as auto, since type(True) is not int).
    was_auto = num_proc is None or type(num_proc) is not int

    if not was_auto:
        # Explicit counts already bypass the small-split guard.
        return _serial_for_the_zoo(get_dataset_num_proc(num_proc, serial_as_none = False))

    rows = _largest_split_rows(trainer)
    if rows is None or rows < ZOO_MIN_ROWS_FOR_MULTIPROC:
        # The escape hatch must win here too.
        env_set, env_value = _from_environment()
        if env_set and env_value is not None:
            return env_value
        # Otherwise it would have gone in-process anyway, and its guard yields
        # None, which is more in-process than the 1 expressible here. Safe whatever
        # the start method: every split here is under the threshold, so its own
        # per-split guard serialises them all.
        return num_proc

    return _serial_for_the_zoo(get_dataset_num_proc(None, serial_as_none = False))

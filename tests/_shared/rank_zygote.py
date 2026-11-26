# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Forks spawned test ranks from a process that has already done the imports a spawned rank would do.

Every gloo test spawns 2 to 8 ranks, and each spawned rank re-imports torch and unsloth: about 9 CPU seconds
of a rank's 11 on a CI runner. With ``UNSLOTH_TEST_RANK_ZYGOTE=1`` (set by distributed-cpu-ci.yml only) a
``multiprocessing`` spawn start in a test process is served instead by a zygote: a stdlib forkserver whose
preload is this module, one per kind of rank, kept per pytest process.

Learning a kind (a rank function plus the short string arguments that pick its case). Its first ranks run as
plain spawns under an observer, which notes the modules unpickling the function imported (the test module's
own imports, which a spawned rank runs before any of its code), what the rank's code imported before the
conftest, and whether it ran ``tests/conftest.py`` at all. Its zygote then imports torch, numpy and pytest,
then those test-module imports, all under the environment a rank starts with. Only when the rank runs the
conftest before importing anything else of its own does the zygote also set the rank's layout variables
(``UNSLOTH_TP_SIZE``, ``UNSLOTH_PARALLEL``, ... from the function's ``env`` argument), run the conftest the
way the harnesses do (``spec_from_file_location`` + ``exec_module``) and import unsloth, recording every
``os.environ`` access.

A forked rank must end up as a spawned rank would have. It starts with the parent's environment and stdio,
the modules the conftest run imported hidden, and the base attributes they changed put back. A variable the
imports ahead read, set while the rank unpickles its function, re-executes it. Nothing more is reused until
the rank executes ``tests/conftest.py``, and there the zygote's result replaces the run only when it cannot
differ:

- the rank has imported none of the hidden modules (unsloth, transformers, ...) on its own,
- every environment variable the conftest run read has the same value in the rank, and every environment
  scan it did would see the same names,
- the base attributes the run changes are still as the zygote found them.

Anything else re-executes the rank as a fresh spawned interpreter (same pid, same pipes), so it runs exactly
as without this module, only later, and is observed again. Consecutive ranks alternate between two zygotes
with different string hash seeds, as spawned ranks each draw their own.
``python tests/_shared/rank_zygote.py --summary DIR`` prints how many ranks took which path.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import sys
import time

ENABLE_ENV = "UNSLOTH_TEST_RANK_ZYGOTE"
STATS_ENV = "UNSLOTH_TEST_RANK_ZYGOTE_STATS"
_SPEC_ENV = "_UNSLOTH_RANK_ZYGOTE_SPEC"
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
CONFTEST = os.path.realpath(os.path.join(_REPO, "tests", "conftest.py"))
_MISSING = object()

# Base: what every gloo test module imports first. A zygote that runs the conftest also imports bitsandbytes
# (the conftest's first import, before its CUDA spoof) and the standard library modules the conftest run pulls
# in (found by a throwaway run in a fork of the zygote) ahead of it, so a rank importing one of those itself
# is no reason to start again.
_BASE_MODULES = ("torch", "torch.distributed", "torch.multiprocessing", "pytest", "numpy")

# Reads of WORLD_SIZE / LOCAL_RANK during the import that cannot change it on a host with no CUDA device
# (the zygote checks torch.cuda.device_count() == 0): (variable, reading function, its caller).
# - bind_local_cuda_device returns before touching anything when CUDA is unavailable.
# - distributed_launch picks the weakest visible card over the current one; with no device both answers
#   are the current-device probe, and _utils' per-card loop runs over zero cards.
_RANK_NEUTRAL_READS = {
    (
        "WORLD_SIZE",
        "unsloth/parallel_env.py:bind_local_cuda_device",
        "unsloth/__init__.py:<module>",
    ),
    (
        "LOCAL_RANK",
        "unsloth/parallel_env.py:bind_local_cuda_device",
        "unsloth/__init__.py:<module>",
    ),
    (
        "WORLD_SIZE",
        "unsloth/device_type.py:distributed_launch",
        "unsloth/device_type.py:cuda_bf16_capability",
    ),
    (
        "WORLD_SIZE",
        "unsloth/device_type.py:distributed_launch",
        "unsloth/models/_utils.py:<module>",
    ),
}
# Read by the CUDA / ROCm runtimes outside os.environ, so compared whatever the recorder saw.
_DEVICE_MASKS = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
# Environment scans during the import that only look at names with a prefix: (scanning function) -> prefix.
_PREFIX_SCANS = {
    "unsloth_zoo/disk_utils.py:is_colab_environment": "COLAB_",
    "tqdm/utils.py:__init__": "TQDM_",
    "networkx/utils/backends.py:_set_configs_from_environment": "NETWORKX_",
    "fsspec/config.py:set_conf_env": "FSSPEC_",
    # {**environ, **out}: a copy handed to refuse_ep_composition, which reads UNSLOTH_* layout names from it.
    "unsloth/parallel_env.py:apply_parallel_env": "UNSLOTH_",
}


def enabled(environ = None) -> bool:
    environ = os.environ if environ is None else environ
    return (
        environ.get(ENABLE_ENV) == "1"
        and environ.get("UNSLOTH_TEST_RANKS_ON_GPU") != "1"
        and sys.platform.startswith("linux")
    )


def _record(kind: str, **fields) -> None:
    """One JSON line per rank event in $UNSLOTH_TEST_RANK_ZYGOTE_STATS/<pid>.jsonl; never raises."""
    root = os.environ.get(STATS_ENV)
    if not root:
        return
    try:
        os.makedirs(root, exist_ok = True)
        with open(os.path.join(root, f"{os.getpid()}.jsonl"), "a", encoding = "utf-8") as f:
            event = {
                "event": kind,
                "pid": os.getpid(),
                "t": time.time(),
                "mem_gb": _mem_available_gb(),
                **fields,
            }
            event.setdefault("test", os.environ.get("PYTEST_CURRENT_TEST"))
            f.write(json.dumps(event) + "\n")
    except Exception:
        pass


# --------------------------------------------------------------------------------------------------
# Parent side: the spawn Popen that asks the zygote instead
# --------------------------------------------------------------------------------------------------

_IN_ZYGOTE = False
_original_popen = None
# Zygotes of this process by the layout variables their import saw, least recently used first.
_servers = {}
_retired = []
_turns = {}  # (layout, mode) -> ranks asked for it: the hash seed alternation, and which zygotes to keep
_tests = {}  # (layout, mode) -> the tests that asked for it
# Tests whose ranks of one kind are spawned as before, ahead of its first zygote.
_HOT = int(os.environ.get("UNSLOTH_TEST_RANK_ZYGOTE_HOT", "1") or "1")
MAX_ZYGOTES_ENV = "UNSLOTH_TEST_RANK_ZYGOTE_MAX"


def install() -> bool:
    """Route this process's multiprocessing spawn starts through the zygote; False when it does not apply."""
    global _original_popen
    import multiprocessing

    if not enabled() or _IN_ZYGOTE or multiprocessing.parent_process() is not None:
        return False  # the pytest process only: not the zygote's own conftest run, not a rank's
    from multiprocessing import popen_spawn_posix

    if _original_popen is None:
        global _OBSERVED, _ObservingPopen
        import tempfile

        _original_popen = popen_spawn_posix.Popen
        # One directory for the whole pytest run, so every xdist worker learns from any worker's ranks.
        run = os.environ.get("PYTEST_XDIST_TESTRUNUID") or str(os.getpid())
        _OBSERVED = os.path.join(tempfile.gettempdir(), f"unsloth_rank_zygote_observed_{run}")
        os.makedirs(_OBSERVED, exist_ok = True)

        class _ObservingPopen(_original_popen):
            """The stdlib spawn, with the process object wrapped so the child notes its imports for us."""

            def __init__(self, process_obj, key):
                self._observed_key = key
                super().__init__(process_obj)

            def _launch(self, process_obj):
                return super()._launch(_Observed(process_obj, self._observed_key))

        popen_spawn_posix.Popen = _popen
    return True


_ObservingPopen = None


class _Observed:
    """Pickles as "start an _Observer, then unpickle the real process object": the observer sees the imports
    the unpickling does, which is where a spawned rank's test module is imported."""

    def __init__(self, process_obj, key):
        self.process_obj, self.key, self._name = process_obj, key, process_obj._name

    def __reduce__(self):
        from multiprocessing.reduction import ForkingPickler
        return (
            _observed_rebuild,
            (_OBSERVED, self.key, bytes(ForkingPickler.dumps(self.process_obj))),
        )


def _observed_rebuild(ranks_file, key, data):
    observer = _Observer(ranks_file, key)
    observer.start()
    process_obj = pickle.loads(data)
    observer.loaded()
    return process_obj


# Per rank function (and the short string arguments that pick its case), learned from a rank that ran as a
# plain spawn under an observer (the first of its kind, or one that had to start again):
# "first": the modules unpickling the function imports (its test module's own imports), which a spawned rank
#   runs before any of its code; the zygote imports them ahead, in the same order.
# "pure": the rank never runs the conftest, loads unsloth files itself, or imports more than the zygote's
#   base before the conftest; the zygote then stops before the conftest and serves only those imports.
_plans = {}
_closures = {}  # rank function -> its test module's imports, from any observed case of it
_cases = {}  # rank function -> {case key: (ran the conftest, unsloth first, imports before the conftest)}
_offsets = {}  # ranks.jsonl path -> bytes read
_GIVE_UP = 6  # starts again with no fork kept: the function's ranks are spawned as before
_WORTH = 300  # modules a zygote that stops before the conftest must import ahead to be worth its own import
_OBSERVED = os.path.join(_HERE, ".")  # placeholder, replaced by install()


def _plan(key):
    return _plans.setdefault(key, {"observed": False, "hits": 0, "restarts": 0})


def _learn():
    paths = [os.path.join(z.dir, "ranks.jsonl") for z in list(_servers.values()) + list(_retired)]
    try:
        paths += [os.path.join(_OBSERVED, n) for n in sorted(os.listdir(_OBSERVED))]
    except OSError:
        pass
    for path in paths:
        try:
            with open(path, encoding = "utf-8") as f:
                f.seek(_offsets.get(path, 0))
                lines = f.readlines()
        except OSError:
            continue
        if lines and not lines[-1].endswith("\n"):
            lines.pop()  # still being written: read it next time
        _offsets[path] = _offsets.get(path, 0) + sum(len(line.encode("utf-8")) for line in lines)
        for line in lines:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            plan = _plan(event.get("target"))
            if event.get("event") in ("hit", "kept"):
                plan["hits"] += 1
            elif event.get("event") == "observed":
                try:
                    target = json.loads(event.get("target"))[0]
                except (TypeError, ValueError, IndexError):
                    target = None
                # The test module's imports are the same for every case of its function.
                _closures[target] = list(event.get("first") or ())
                _cases.setdefault(target, {})[event.get("target")] = (
                    bool(event.get("conftest")),
                    bool(event.get("unsloth_first")),
                    list(event.get("before_conftest") or ()),
                )
                plan.update(
                    observed = True,
                    first = list(event.get("first") or ()),
                    before_conftest = list(event.get("before_conftest") or ()),
                    conftest = bool(event.get("conftest")),
                    unsloth_first = bool(event.get("unsloth_first")),
                )
            else:
                plan["restarts"] += 1


def _pure(target) -> bool:
    """Whether a function's zygote stops before the conftest: unless every case of it seen so far ran the
    conftest before importing anything of its own (the zygote's base aside) or loading unsloth itself."""
    # Known from a running zygote; before the first one, what every zygote imports ahead whatever else.
    base = _base_tops() or {"torch", "numpy", "pytest", "_pytest", "pluggy", "bitsandbytes"}
    cases = _cases.get(target) or {}
    return not cases or any(
        not conftest or unsloth_first or any(n.split(".", 1)[0] not in base for n in before)
        for conftest, unsloth_first, before in cases.values()
    )


def _base_tops():
    for zygote in _servers.values():
        tops = zygote.status().get("base_tops")
        if tops is not None:
            return set(tops)
    return None


def _popen(process_obj):
    if not enabled():
        return _original_popen(process_obj)
    import threading

    global _LOCK
    if _LOCK is None:
        _LOCK = threading.RLock()
    with _LOCK:  # ranks started from several threads share the plans and the zygotes
        return _popen_locked(process_obj)


_LOCK = None


def _popen_locked(process_obj):
    _learn()
    key = _plan_key(process_obj)
    plan = _plan(key)
    closure = _closures.get(_target_name(process_obj))
    if closure is None:
        # Not seen yet: a plain spawn, watched, says what its ranks import.
        _record("spawn", target = key, why = "observed")
        return _ObservingPopen(process_obj, key)
    if "torch" not in closure:
        _record("spawn", target = key, why = "torch is left to the rank's code")
        return _original_popen(process_obj)
    if plan["restarts"] >= _GIVE_UP and plan["hits"] == 0:
        _record("spawn", target = key, why = "its forks kept starting again")
        return _original_popen(process_obj)
    # A function whose cases all ran the conftest first reuses a zygote's conftest run (a case that does
    # otherwise starts again at the conftest, before its own code goes further); any other function forks
    # from a zygote that stops before the conftest and serves only its test module's imports.
    mode = {"first": closure, "pure": _pure(_target_name(process_obj))}
    if mode["pure"] and len(closure) < _WORTH:
        _record("spawn", target = key, why = "too little to import ahead")
        return _original_popen(process_obj)
    zygote = _zygote_for(process_obj, mode)
    if zygote is None or zygote.status().get("broken"):
        why = "no zygote for this kind yet" if zygote is None else "zygote broken"
        _record("spawn", target = key, why = why)
        return _original_popen(process_obj)
    differs = _early_differs(zygote)
    if differs:
        # The zygote imported ahead under another value (a test sets it per test): a fork would only start
        # again at once.
        _record("spawn", target = key, why = f"env {differs} differs from the zygote's")
        return _original_popen(process_obj)
    return _ZygotePopen(process_obj, zygote, key)


def _early_differs(zygote):
    """A variable the zygote's imports ahead read whose value the rank would not start with, or None."""
    early = zygote.status().get("early_inputs")
    if early is None:
        return None  # still importing: the rank itself checks when it starts
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    for key, value in early.items():
        if env.get(key) != value and key not in _DEVICE_MASKS:
            return key
    return None


def _plan_key(process_obj) -> str:
    """The rank function, plus the short identifier-like string arguments that choose its case."""
    args = list(getattr(process_obj, "_args", ()) or ())
    if getattr(getattr(process_obj, "_target", None), "__name__", "") == "_wrap" and len(args) >= 3:
        args = list(args[2]) if isinstance(args[2], (list, tuple)) else []
    picks = [a for a in args if isinstance(a, str) and a.isidentifier() and len(a) <= 40]
    return json.dumps([_target_name(process_obj), picks])


def _env_args(process_obj) -> dict:
    """Environment the rank will apply before importing, as far as its arguments tell: every str -> str dict of
    upper-case names (the harnesses' ``env`` argument). Only a guess; the rank checks the real one."""
    found = {}
    values = list(getattr(process_obj, "_args", ()) or ()) + list(
        (getattr(process_obj, "_kwargs", {}) or {}).values()
    )
    for value in values:
        if isinstance(value, (list, tuple)):
            values.extend(v for v in value if isinstance(v, (dict, list, tuple)))
        elif (
            isinstance(value, dict)
            and value
            and all(
                isinstance(k, str) and k.isupper() and isinstance(v, str) for k, v in value.items()
            )
        ):
            found.update(value)
    return found


_WORLD_PARAMS = ("world", "world_size", "nprocs", "n_ranks", "ranks")


def _world_arg(process_obj):
    """The world size a harness passes its rank function (a parameter named world / world_size / ...), or None.
    torch.multiprocessing.spawn's ``_wrap(fn, i, args, error_file)`` is looked through to ``fn(i, *args)``."""
    import inspect

    target = getattr(process_obj, "_target", None)
    args = tuple(getattr(process_obj, "_args", ()) or ())
    kwargs = dict(getattr(process_obj, "_kwargs", {}) or {})
    if getattr(target, "__name__", "") == "_wrap" and len(args) >= 3 and callable(args[0]):
        target, args, kwargs = args[0], (args[1], *tuple(args[2])), {}
    try:
        bound = inspect.signature(target).bind_partial(*args, **kwargs).arguments
    except (TypeError, ValueError):
        return None
    for name in _WORLD_PARAMS:
        value = bound.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _zygote_for(process_obj, mode):
    """The zygote whose import environment matches what this rank is expected to have, started if needed."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""  # what every gloo test module sets in its ranks before torch
    guess = _env_args(process_obj)
    inputs = None
    for zygote in _servers.values():
        inputs = zygote.status().get("env_input_names")
        if inputs is not None:
            break
    if inputs is not None:
        # Only variables the import reads tell zygotes apart; the rest cannot change what it builds.
        guess = {k: v for k, v in guess.items() if k in inputs}
    layout = [list(kv) for kv in sorted(guess.items())]
    known = next(
        (z for k, z in _servers.items() if json.loads(k)[:2] == [layout, mode] and z.status()), None
    )
    world = _world_arg(process_obj)
    if known is not None and world is not None and known.status().get("rank_inputs"):
        # This layout's import reads WORLD_SIZE where it matters (UNSLOTH_PARALLEL translates by it).
        guess["WORLD_SIZE"] = str(world)
        layout = [list(kv) for kv in sorted(guess.items())]
    if mode["pure"]:
        guess = {}  # it stops before the conftest, the only import that reads the layout
        layout = []
    elif any(os.sep in v for v in guess.values()):
        return None  # a path the import reads (a per-test cache directory): no other rank would share it
    # Spawned ranks each draw their own string hash seed, so a set iterated in a different order on two
    # ranks shows. A fork keeps its zygote's seed: consecutive ranks alternate between two zygotes with
    # different seeds, unless PYTHONHASHSEED fixes one seed for every process anyway.
    fixed = os.environ.get("PYTHONHASHSEED", "random") != "random"
    turns = json.dumps([layout, mode])
    turn = _turns[turns] = _turns.get(turns, -1) + 1
    tests = _tests.setdefault(turns, set())
    tests.add(os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" ", 1)[0])
    key = json.dumps([layout, mode, 0 if fixed else turn % 2])
    zygote = _servers.pop(key, None)
    if zygote is None:
        if len(tests) <= _HOT:
            # A zygote costs one import of its own: only for a kind of rank that a second test asks for.
            # Counted in tests, so the ranks of one test all start the same way and meet at the same time.
            return None
        limit = _limit()
        if _mem_available_gb() < _MEM_FLOOR_GB:
            limit = 0  # memory is short: every idle zygote goes, and no new one starts
        # Least used first, among those with no live rank (a rank's exit code comes back through its zygote).
        for old_key in sorted(_servers, key = lambda k: _turns.get(json.dumps(json.loads(k)[:2]), 0)):
            if len(_servers) < limit:
                break
            if _servers[old_key].idle():
                old = _servers.pop(old_key)
                old.stop()
                _retired.append(old)
        if len(_servers) >= limit:
            return None  # every zygote still has a live rank: a plain spawn
        zygote = _Zygote(env, mode, guess)
        twin = json.dumps([layout, mode, 1 - json.loads(key)[2]])
        if not fixed and twin not in _servers and len(_servers) + 1 < limit:
            # Its other-seed twin starts now too, so the next rank of this test does not wait for an import.
            _servers[twin] = _Zygote(env, mode, guess)
    _servers[key] = zygote
    return zygote


_MEM_FLOOR_GB = float(os.environ.get("UNSLOTH_TEST_RANK_ZYGOTE_MIN_FREE_GB", "4") or "4")


def _mem_available_gb() -> float:
    """MemAvailable from /proc/meminfo, in GB (inf when unreadable): a zygote is a whole import."""
    try:
        with open("/proc/meminfo", encoding = "utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
    except (OSError, ValueError, IndexError):
        pass
    return float("inf")


def _limit() -> int:
    """Zygotes per pytest process: each holds a whole import (0.5 to 0.8 GB), so 8 over the xdist workers."""
    if os.environ.get(MAX_ZYGOTES_ENV):
        return max(1, int(os.environ[MAX_ZYGOTES_ENV]))
    workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1") or "1")
    return max(2, 8 // max(1, workers))


class _Zygote:
    """One forkserver whose preload imported unsloth under ``env``."""

    _count = 0

    def __init__(self, env, mode, layout):
        import tempfile
        import weakref
        from multiprocessing import forkserver

        _Zygote._count += 1
        self.dir = os.path.join(
            tempfile.gettempdir(), f"unsloth_rank_zygote_{os.getpid()}_{_Zygote._count}"
        )
        os.makedirs(self.dir, exist_ok = True)
        self.ranks = weakref.WeakSet()
        spec = {
            "env": env,
            "sys_path": list(sys.path),
            "cwd": os.getcwd(),
            "status": os.path.join(self.dir, "status.json"),
            # No device the parent can see: a missing and an empty device mask then mean the same.
            "no_gpu_host": _no_gpu_host(),
            # Another zygote's throwaway import already listed them: the same process, the same list.
            "stdlib": next(
                (
                    z.status()["stdlib"]
                    for z in _servers.values()
                    if "stdlib" in z.status() and not z.status().get("pure")
                ),
                None,
            ),
            "first": mode["first"],
            "pure": mode["pure"],
            "layout": layout,
        }
        spec_path = os.path.join(self.dir, "spec.json")
        with open(spec_path, "w", encoding = "utf-8") as f:
            json.dump(spec, f)
        self.server = forkserver.ForkServer()
        self.server.set_forkserver_preload(["rank_zygote"])
        saved = {k: os.environ.get(k) for k in (_SPEC_ENV, "PYTHONPATH", "PYTHONHASHSEED")}
        if saved["PYTHONHASHSEED"] in (None, "random"):
            # Read once at interpreter start (spec["env"] is applied after): this zygote's own random seed.
            os.environ["PYTHONHASHSEED"] = str(
                int.from_bytes(os.urandom(4), "little") % 4294967295 + 1
            )
        os.environ[_SPEC_ENV] = spec_path
        os.environ["PYTHONPATH"] = os.pathsep.join(p for p in (_HERE, saved["PYTHONPATH"]) if p)
        try:
            # The server inherits this environment when it starts (its preload sets spec["env"]); restored after.
            self.server.ensure_running()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self._status = None

    def status(self) -> dict:
        if self._status is None:
            try:
                with open(os.path.join(self.dir, "status.json"), encoding = "utf-8") as f:
                    self._status = json.load(f)
            except (OSError, ValueError):
                return {}  # still importing
        return self._status

    def idle(self) -> bool:
        # A rank's exit code comes back through its zygote, so one with a live rank must keep running.
        return all(p.poll() is not None for p in list(self.ranks))

    def stop(self):
        try:
            self.server._stop()
        except Exception:
            pass


try:
    from multiprocessing import popen_forkserver as _pf
except ImportError:  # no fd passing on this platform
    _pf = None


if _pf is not None:

    class _ZygotePopen(_pf.Popen):
        """popen_forkserver.Popen against a zygote, with a header the rank applies first."""

        method = "spawn"  # what the caller asked for; the context checks compare against it

        def __init__(self, process_obj, zygote, key):
            self._zygote, self._key = zygote, key
            super().__init__(process_obj)

        def _launch(self, process_obj):
            from multiprocessing import forkserver, spawn, util
            from multiprocessing.context import reduction, set_spawning_popen

            # The current stdout / stderr first: pytest's capture of the running test, as a spawned rank has.
            self._fds = [1, 2]
            prep_data = spawn.get_preparation_data(process_obj._name)
            buf = io.BytesIO()
            set_spawning_popen(self)
            try:
                reduction.dump(prep_data, buf)
                reduction.dump(process_obj, buf)
            finally:
                set_spawning_popen(None)
            header = {
                "env": dict(os.environ),
                "stdio": (0, 1),
                "size": buf.getbuffer().nbytes,
                "target": self._key,
            }
            self.sentinel, w = self._zygote.server.connect_to_new_process(self._fds)
            _parent_w = os.dup(w)
            self.finalizer = util.Finalize(self, util.close_fds, (_parent_w, self.sentinel))
            with open(w, "wb", closefd = True) as f:
                pickle.dump(header, f, protocol = pickle.HIGHEST_PROTOCOL)
                f.write(buf.getbuffer())
            self.pid = forkserver.read_signed(self.sentinel)
            self._zygote.ranks.add(self)
            _record(
                "start",
                rank_pid = self.pid,
                target = _target_name(process_obj),
                zygote = self._zygote.dir,
            )

else:  # pragma: no cover
    _ZygotePopen = None


def _no_gpu_host() -> bool:
    try:
        import torch
        return torch.cuda.device_count() == 0 and not getattr(torch.version, "hip", None)
    except Exception:
        return False


def _target_name(process_obj) -> str:
    target = getattr(process_obj, "_target", None)
    args = tuple(getattr(process_obj, "_args", ()) or ())
    if getattr(target, "__name__", "") == "_wrap" and args and callable(args[0]):
        target = args[0]  # torch.multiprocessing.spawn's wrapper: name the rank function
    return f"{getattr(target, '__module__', '?')}.{getattr(target, '__qualname__', repr(target))}"


# --------------------------------------------------------------------------------------------------
# Zygote side: preload (runs in the forkserver process when it imports this module)
# --------------------------------------------------------------------------------------------------

_Z = None  # the zygote's record, inherited by every forked rank


class _EnvRecorder:
    """Every os.environ access while active: the first access per name, and who read it."""

    def __init__(self):
        self.first = {}  # name -> ("read", value or None) | ("write", None)
        self.readers = {}  # name -> {(reader, caller)}
        self.scans = set()  # functions that iterated the whole environment
        self._depth = 0
        self._saved = None

    @staticmethod
    def _site(depth):
        f = sys._getframe(depth)
        skip = ("/os.py", "_collections_abc.py", "<frozen", "rank_zygote.py")
        sites = []
        while f is not None and len(sites) < 2:
            name = f.f_code.co_filename
            if not any(s in name for s in skip):
                sites.append(f"{_short(name)}:{f.f_code.co_name}")
            f = f.f_back
        while len(sites) < 2:
            sites.append("?")
        return tuple(sites)

    def start(self):
        E = type(os.environ)
        self._saved = (E.__getitem__, E.__setitem__, E.__delitem__, E.__contains__, E.__iter__)
        get, set_, del_, has, it = self._saved
        rec = self

        def note_read(key, value):
            if rec._depth:
                return
            rec.first.setdefault(key, ("read", value))
            rec.readers.setdefault(key, set()).add(rec._site(1))

        def __getitem__(self, key):
            try:
                value = get(self, key)
            except KeyError:
                note_read(key, None)
                raise
            note_read(key, value)
            return value

        def __contains__(self, key):
            present = has(self, key)
            note_read(key, get(self, key) if present else None)
            return present

        def __setitem__(self, key, value):
            rec.first.setdefault(key, ("write", None))
            return set_(self, key, value)

        def __delitem__(self, key):
            rec.first.setdefault(key, ("write", None))
            return del_(self, key)

        def __iter__(self):
            rec.scans.add(rec._site(1)[0])
            rec._depth += 1
            try:
                yield from it(self)
            finally:
                rec._depth -= 1

        E.__getitem__, E.__setitem__, E.__delitem__ = __getitem__, __setitem__, __delitem__
        E.__contains__, E.__iter__ = __contains__, __iter__

    def stop(self):
        E = type(os.environ)
        E.__getitem__, E.__setitem__, E.__delitem__, E.__contains__, E.__iter__ = self._saved


def _short(filename: str) -> str:
    """unsloth/x.py, unsloth_zoo/x.py, tests/x.py or site-packages-relative, as the allow lists name them."""
    filename = filename.replace(os.sep, "/")
    for marker in ("/site-packages/", "/dist-packages/"):
        if marker in filename:
            return filename.split(marker, 1)[1]
    for top in ("/unsloth_zoo/", "/unsloth/", "/tests/"):
        if top in filename:
            return top[1:] + filename.rsplit(top, 1)[1]
    return filename


def _owners():
    """Every base module and every class reachable from one: whose attributes the import may rebind."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # isinstance on deprecated module attributes warns
        return _owners_quiet()


def _owners_quiet():
    owners, seen = [], set()
    for name, mod in list(sys.modules.items()):
        if mod is None or id(mod) in seen:
            continue
        seen.add(id(mod))
        owners.append(mod)
        try:
            values = list(vars(mod).values())
        except TypeError:
            continue
        for value in values:
            if isinstance(value, type) and id(value) not in seen:
                seen.add(id(value))
                owners.append(value)
    return owners


def _attrs(owner):
    try:
        return dict(vars(owner))
    except TypeError:
        return {}


def _set_attr(owner, name, value):
    try:
        if value is _MISSING:
            if name in vars(owner):
                delattr(owner, name) if isinstance(owner, type) else vars(owner).pop(name, None)
        elif isinstance(owner, type):
            setattr(owner, name, value)
        else:
            vars(owner)[name] = value
    except (AttributeError, TypeError):
        pass


def _globals_snapshot():
    import warnings
    return {
        "warnings": list(warnings.filters),
        "meta_path": list(sys.meta_path),
        "path_hooks": list(sys.path_hooks),
    }


def _apply_globals(snap):
    import warnings

    warnings.filters[:] = snap["warnings"]
    if hasattr(warnings, "_filters_mutated"):
        warnings._filters_mutated()
    sys.meta_path[:] = snap["meta_path"]
    sys.path_hooks[:] = snap["path_hooks"]
    sys.path_importer_cache.clear()


def _stdlib_imported_by_the_conftest() -> list:
    """Standard library modules a conftest run + ``import unsloth`` adds, in import order, from a throwaway fork."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            os.close(r)
            import importlib.util

            before = set(sys.modules)
            conf_spec = importlib.util.spec_from_file_location("_tp_conftest", CONFTEST)
            conf_spec.loader.exec_module(importlib.util.module_from_spec(conf_spec))
            import unsloth  # noqa: F401

            names = [
                n
                for n in list(sys.modules)
                if n not in before and n.split(".", 1)[0] in sys.stdlib_module_names
            ]
            with os.fdopen(w, "w", encoding = "utf-8") as f:
                json.dump(names, f)
            code = 0
        finally:
            os._exit(code)
    os.close(w)
    with os.fdopen(r, encoding = "utf-8") as f:
        data = f.read()
    _, sts = os.waitpid(pid, 0)
    if os.waitstatus_to_exitcode(sts) != 0:
        raise RuntimeError("the throwaway import failed")
    return json.loads(data)


def _preload(spec_path: str) -> None:
    global _Z, _IN_ZYGOTE
    _IN_ZYGOTE = True
    with open(spec_path, encoding = "utf-8") as f:
        spec = json.load(f)
    status = {"broken": None}
    try:
        os.environ.clear()
        os.environ.update(spec["env"])
        sys.path[:] = spec["sys_path"]
        os.chdir(spec["cwd"])
        base_recorder = _EnvRecorder()
        base_recorder.start()
        try:
            for name in _BASE_MODULES:
                try:
                    __import__(name)
                except ImportError:
                    pass
        finally:
            base_recorder.stop()
        stdlib = spec.get("stdlib")
        if stdlib is None:
            # Only a zygote that runs the conftest imports these ahead.
            stdlib = [] if spec.get("pure") else _stdlib_imported_by_the_conftest()
        # The test module's own imports, under the environment a rank has when it unpickles its function.
        closure_recorder = _EnvRecorder()
        closure_recorder.start()
        try:
            for name in spec.get("first", ()):
                try:
                    __import__(name)
                except Exception:
                    pass
        finally:
            closure_recorder.stop()
        # The rank's layout variables, set by its function before the conftest; then what the conftest
        # imports first, unless this zygote stops before the conftest.
        os.environ.update(spec.get("layout") or {})
        late_recorder = _EnvRecorder()
        late_recorder.start()
        try:
            for name in () if spec.get("pure") else ("bitsandbytes", *stdlib):
                try:
                    __import__(name)
                except Exception:
                    pass
        finally:
            late_recorder.stop()
        base = {name for name, mod in sys.modules.items() if mod is not None}
        owners = _owners()
        before = [(o, _attrs(o)) for o in owners]
        globals_before = _globals_snapshot()
        env_before = dict(os.environ)

        import importlib.util

        recorder = _EnvRecorder()
        conftest = None
        if not spec.get("pure"):
            recorder.start()
            try:
                conf_spec = importlib.util.spec_from_file_location("_tp_conftest", CONFTEST)
                conftest = importlib.util.module_from_spec(conf_spec)
                conf_spec.loader.exec_module(conftest)
                import unsloth  # noqa: F401  (the conftest swallows a refusal; this one must not)
            finally:
                recorder.stop()
        env_after = dict(os.environ)

        changes = []
        for owner, old in before:
            new = _attrs(owner)
            for key in set(old) | set(new):
                a, b = old.get(key, _MISSING), new.get(key, _MISSING)
                if a is not b:
                    changes.append((owner, key, a, b))
        hidden = {
            name: mod for name, mod in sys.modules.items() if name not in base and mod is not None
        }
        import torch

        late_inputs = {k: v for k, (kind, v) in late_recorder.first.items() if kind == "read"}
        early_inputs = {
            **{k: v for k, (kind, v) in closure_recorder.first.items() if kind == "read"},
            **{k: v for k, (kind, v) in base_recorder.first.items() if kind == "read"},
        }
        # torch, numpy and pytest come first in a rank too (its test module imports them before running the
        # conftest); the late ones a rank imports during the conftest, after the variables it sets.
        written = sorted(
            set(late_inputs) & {k for k, (kind, _) in recorder.first.items() if kind == "write"}
        )
        if written:
            raise RuntimeError(
                f"the imports before the conftest read {written}, which the conftest writes"
            )
        _Z = {
            "spec": spec,
            "conftest_env": env_before,
            "hidden": hidden,
            "changes": changes,
            "globals_before": globals_before,
            "globals_after": _globals_snapshot(),
            "early_env_inputs": early_inputs,
            "late_env_inputs": late_inputs,
            "env_inputs": {
                **{k: spec["env"].get(k) for k in _DEVICE_MASKS},
                **{k: v for k, (kind, v) in recorder.first.items() if kind == "read"},
            },
            "env_readers": recorder.readers,
            "env_scans": recorder.scans,
            "env_writes": {
                k: env_after.get(k)
                for k in set(env_before) | set(env_after)
                if env_before.get(k) != env_after.get(k)
            },
            "conftest": None if conftest is None else dict(vars(conftest)),
            "rank_neutral": torch.cuda.device_count() == 0,
        }
        unknown_scans = sorted(s for s in recorder.scans if s not in _PREFIX_SCANS)
        status.update(
            early_inputs = early_inputs,
            base_tops = sorted({name.split(".", 1)[0] for name in base}),
            first = list(spec.get("first", ())),
            pure = bool(spec.get("pure")),
            stdlib = stdlib,
            env_input_names = sorted(_Z["env_inputs"]),
            # WORLD_SIZE / LOCAL_RANK read where the value can change the import: zygotes then differ by it.
            rank_inputs = sorted(
                key
                for key in ("WORLD_SIZE", "LOCAL_RANK")
                if any(
                    (key, reader, caller) not in _RANK_NEUTRAL_READS
                    for reader, caller in recorder.readers.get(key, ())
                )
            ),
            hidden = len(hidden),
            changes = len(changes),
            env_inputs = len(_Z["env_inputs"]),
            unknown_scans = unknown_scans,
            rank_neutral = _Z["rank_neutral"],
        )
        import threading

        # Native pools (OpenBLAS, jemalloc) are rebuilt after a fork, as for a DataLoader worker; a Python
        # thread could hold a lock the forked rank then waits on forever, so one means no zygote.
        status["native_threads"] = len(os.listdir("/proc/self/task"))
        if threading.active_count() != 1:
            status["broken"] = (
                f"Python threads after the import: {[t.name for t in threading.enumerate()]}"
            )
            _Z = None
    except BaseException as e:  # noqa: BLE001 - any failure means plain spawns, reported in status
        status["broken"] = f"{type(e).__name__}: {e}"
        _Z = None
    try:
        with open(spec["status"], "w", encoding = "utf-8") as f:
            json.dump(status, f)
    except OSError:
        pass
    from multiprocessing import spawn

    spawn._main = _rank_main


# --------------------------------------------------------------------------------------------------
# Rank side: runs in each forked rank
# --------------------------------------------------------------------------------------------------

_rank = None  # {"data": bytes, "env": dict, "sentinel": fd, "decided": None | "hit" | reason}


def _rank_main(fd, parent_sentinel):
    """spawn._main for a forked rank: header, then the spawn payload, read whole so a re-exec can replay it."""
    global _rank
    from multiprocessing import process, spawn

    with os.fdopen(fd, "rb", closefd = True) as from_parent:
        header = pickle.load(from_parent)
        data = from_parent.read(header["size"])
    from multiprocessing import forkserver

    inherited = list(forkserver._forkserver._inherited_fds or [])
    stdout_ind, stderr_ind = header["stdio"]
    os.dup2(inherited[stdout_ind], 1)
    os.dup2(inherited[stderr_ind], 2)
    os.environ.clear()
    os.environ.update(header["env"])
    _rank = {
        "data": data,
        "env": dict(header["env"]),
        "sentinel": parent_sentinel,
        "decided": None,
        "target": header.get("target"),
    }
    if _Z is None:
        _respawn("no zygote")
    frozen = dict(_Z["early_env_inputs"])
    for key, value in frozen.items():
        if os.environ.get(key) != value:
            _respawn(f"env {key}, which the imports ahead read")
    # torch / numpy / pytest and the test module's own imports already ran, under these values. A spawned rank
    # runs them while it unpickles its function, so a variable the test module sets before them restarts the
    # rank; once it is loaded they are done either way. What the conftest imports is checked at the conftest.
    _rank["frozen"] = frozen
    _guard_env()
    _hide()

    stream = io.BytesIO(data)
    process.current_process()._inheriting = True
    try:
        preparation_data = pickle.load(stream)
        spawn.prepare(preparation_data)
        self = pickle.load(stream)
    finally:
        del process.current_process()._inheriting
    target = getattr(self, "_target", None)
    if getattr(target, "__name__", "") == "_wrap" and getattr(self, "_args", None):
        target = self._args[0]  # torch.multiprocessing.spawn's wrapper
    target_module = sys.modules.get(getattr(target, "__module__", None))
    if target_module is None or not isinstance(vars(target_module).get("torch"), type(sys)):
        # torch comes later in this rank (its code imports it, after whatever it set first): start again
        # now, before any of that code runs.
        _respawn("the test module does not import torch itself")
    _rank["frozen"] = None
    code = self._bootstrap(parent_sentinel)
    _exit_like_a_spawned_rank()
    return code


def _exit_like_a_spawned_rank():
    """A spawned rank leaves through sys.exit: interpreter shutdown waits for its non-daemon threads, then runs
    its atexit handlers. The forkserver ends a forked rank with os._exit as soon as this returns, so both happen
    here first. (Finalizing modules and objects is the one step a forked rank still skips.)"""
    import atexit
    import threading

    try:
        threading._shutdown()
    except Exception:
        pass
    atexit._run_exitfuncs()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass


def _guard_env():
    """Re-execute the rank when it sets a variable the zygote's imports before the conftest read."""
    E = type(os.environ)
    setitem, delitem = E.__setitem__, E.__delitem__

    def check(key, value):
        frozen = _rank.get("frozen")
        if frozen and key in frozen and frozen[key] != value:
            _respawn(f"env {key} set by the rank, which the imports before the conftest read")

    def __setitem__(self, key, value):
        if self is os.environ:
            check(key, value)
        return setitem(self, key, value)

    def __delitem__(self, key):
        if self is os.environ:
            check(key, None)
        return delitem(self, key)

    E.__setitem__, E.__delitem__ = __setitem__, __delitem__


class _HiddenFinder:
    """Sees the first import of a hidden module: the rank imports on its own, so it re-executes."""

    @staticmethod
    def find_spec(
        name,
        path = None,
        target = None,
    ):
        if _rank is not None and _rank["decided"] is None and name in _Z["hidden"]:
            _respawn(f"imported {name} before the conftest")
        return None


def _hide():
    """The rank's start: hidden modules out of sys.modules, base attributes as before the import."""
    import importlib._bootstrap_external as ext

    for name in _Z["hidden"]:
        sys.modules.pop(name, None)
    for owner, key, old, _new in _Z["changes"]:
        _set_attr(owner, key, old)
    _apply_globals(_Z["globals_before"])
    sys.meta_path.insert(0, _HiddenFinder)
    original = ext.SourceFileLoader.exec_module

    def exec_module(self, module):
        if (
            _Z["conftest"] is not None
            and _rank["decided"] is None
            and _is_conftest(getattr(module, "__file__", None))
        ):
            _decide(module)
            return None
        return original(self, module)

    exec_module._rank_zygote_original = original
    ext.SourceFileLoader.exec_module = exec_module


def _is_conftest(path):
    try:
        return path is not None and os.path.realpath(path) == CONFTEST
    except (TypeError, ValueError):
        return False


def _why_not(module) -> str | None:
    """None when the zygote's conftest run is what this rank's would be, else the first reason it is not."""
    for name in _Z["hidden"]:
        if name in sys.modules:
            return f"{name} already imported"
    for name in sys.modules:
        if name == "unsloth" or name.startswith(("unsloth.", "unsloth_zoo")):
            return f"{name} already imported"
    env = os.environ
    for key, value in _Z["late_env_inputs"].items():
        if env.get(key) != value:
            return f"env {key}, which the imports before the conftest read"
    for key, value in _Z["env_inputs"].items():
        if env.get(key) == value:
            continue
        if (
            key in _DEVICE_MASKS
            and _Z["spec"]["no_gpu_host"]
            and env.get(key) in (None, "")
            and value in (None, "")
        ):
            continue
        if _Z["rank_neutral"] and all(
            (key, reader, caller) in _RANK_NEUTRAL_READS
            for reader, caller in _Z["env_readers"].get(key, ())
        ):
            continue
        return f"env {key}"
    for scan in _Z["env_scans"]:
        prefix = _PREFIX_SCANS.get(scan)
        if prefix is None:
            if dict(env) != _Z["conftest_env"]:
                return f"environment scanned by {scan}"
            continue
        ours = {k: v for k, v in env.items() if k.startswith(prefix)}
        theirs = {k: v for k, v in _Z["conftest_env"].items() if k.startswith(prefix)}
        if ours != theirs:
            return f"env {prefix}*"
    before = _Z["globals_before"]
    if [f for f in sys.meta_path if f is not _HiddenFinder] != before["meta_path"]:
        return "sys.meta_path changed before the conftest"
    if sys.path_hooks != before["path_hooks"]:
        return "sys.path_hooks changed before the conftest"
    for owner, key, old, _new in _Z["changes"]:
        if _attrs(owner).get(key, _MISSING) is not old:
            return f"{getattr(owner, '__name__', owner)}.{key} changed before the conftest"
    return None


def _decide(module):
    reason = _why_not(module)
    if reason is not None:
        _respawn(reason)
    _rank["decided"] = "hit"
    _rank["frozen"] = None  # the rank is where the conftest leaves it, as a spawned one would be
    import importlib._bootstrap_external as ext

    ext.SourceFileLoader.exec_module = ext.SourceFileLoader.exec_module._rank_zygote_original
    sys.meta_path[:] = [f for f in sys.meta_path if f is not _HiddenFinder]
    sys.modules.update(_Z["hidden"])
    for owner, key, _old, new in _Z["changes"]:
        _set_attr(owner, key, new)
    import warnings

    after, before = _Z["globals_after"], _Z["globals_before"]
    sys.meta_path[:] = after["meta_path"]
    sys.path_hooks[:] = after["path_hooks"]
    sys.path_importer_cache.clear()
    # The conftest's filters go in front of the rank's own, as they would have been added after them.
    warnings.filters[:0] = [f for f in after["warnings"] if f not in before["warnings"]]
    if hasattr(warnings, "_filters_mutated"):
        warnings._filters_mutated()
    for key, value in _Z["env_writes"].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    keep = {
        "__name__",
        "__file__",
        "__spec__",
        "__loader__",
        "__cached__",
        "__package__",
        "__builtins__",
    }
    vars(module).update({k: v for k, v in _Z["conftest"].items() if k not in keep})
    _record("hit")
    _tell_parent("hit")


def _tell_parent(event, reason = None):
    """What this rank did, for the parent's choice of path for the next rank of the same function."""
    try:
        with open(
            os.path.join(os.path.dirname(_Z["spec"]["status"]), "ranks.jsonl"),
            "a",
            encoding = "utf-8",
        ) as f:
            f.write(
                json.dumps({"target": _rank["target"], "event": event, "reason": reason}) + "\n"
            )
    except Exception:
        pass


def _respawn(reason: str):
    """Re-execute this rank as a fresh spawned interpreter: same pid, same pipes, the parent's environment."""
    from multiprocessing import forkserver, resource_tracker, util

    _record("respawn", reason = reason)
    _tell_parent("respawn", reason)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    data_fd = os.memfd_create("rank_zygote_payload", 0)
    os.write(data_fd, _rank["data"])
    os.lseek(data_fd, 0, os.SEEK_SET)
    inherited = list(forkserver._forkserver._inherited_fds or [])
    tracker = resource_tracker._resource_tracker._fd
    keep = [data_fd, _rank["sentinel"], *inherited] + ([tracker] if tracker is not None else [])
    for fd in keep:
        os.set_inheritable(fd, True)
    args = json.dumps(
        {
            "data": data_fd,
            "sentinel": _rank["sentinel"],
            "inherited": inherited,
            "tracker": tracker,
            "ranks": os.path.join(os.path.dirname(_Z["spec"]["status"]), "ranks.jsonl")
            if _Z is not None
            else None,
            "target": _rank["target"],
        }
    )
    code = (
        f"import sys; sys.path.insert(0, {_HERE!r}); import rank_zygote; "
        f"sys.path.remove({_HERE!r}); rank_zygote._respawned({args!r})"
    )
    exe = sys.executable
    os.execve(exe, [exe, *util._args_from_interpreter_flags(), "-c", code], _rank["env"])


def _respawned(args_json: str):
    """The re-executed rank: what spawn_main does, on the payload the zygote received."""
    from multiprocessing import forkserver, process, resource_tracker, spawn

    args = json.loads(args_json)
    forkserver._forkserver._inherited_fds = args["inherited"]
    resource_tracker._resource_tracker._fd = args["tracker"]
    if not args.get("ranks"):
        sys.exit(spawn._main(args["data"], args["sentinel"]))
    # spawn._main, with the modules the payload's own import pulls in noted for the parent.
    seen = _Observer(args["ranks"], args["target"])
    with os.fdopen(args["data"], "rb", closefd = True) as from_parent:
        process.current_process()._inheriting = True
        try:
            spawn.prepare(pickle.load(from_parent))
            seen.start()
            self = pickle.load(from_parent)
            seen.loaded()
        finally:
            del process.current_process()._inheriting
    sys.exit(self._bootstrap(args["sentinel"]))


class _Observer:
    """In a plain-spawned or re-executed rank, only watching: which modules unpickling its function imported (its
    test module's own imports, which a spawned rank runs before any of its code), what its code imported before
    the conftest, whether it ran the conftest, and whether it loaded unsloth modules itself first. Reported once,
    when the conftest starts or at exit, for the parent's next zygote for this function."""

    def __init__(self, ranks_file, key):
        self.ranks_file, self.key = ranks_file, key
        self.first, self.before_conftest = None, []
        self.conftest = self.unsloth_first = self.reported = False

    def start(self):
        import atexit
        import importlib._bootstrap_external as ext

        self.before = set(sys.modules)
        sys.meta_path.insert(0, self)
        original = ext.SourceFileLoader.exec_module
        observer = self

        def exec_module(loader, module):
            if not observer.conftest and _is_conftest(getattr(module, "__file__", None)):
                observer.conftest = True
                observer.report()
            return original(loader, module)

        ext.SourceFileLoader.exec_module = exec_module
        atexit.register(self.report)

    def find_spec(
        self,
        name,
        path = None,
        target = None,
    ):
        if self.conftest:
            return None
        top = name.split(".", 1)[0]
        if top in ("unsloth", "unsloth_zoo"):
            self.unsloth_first = True
        elif self.first is not None and top not in sys.stdlib_module_names and top != "tests":
            self.before_conftest.append(name)
        return None

    def loaded(self):
        self.first = []
        for name in list(sys.modules):
            top = name.split(".", 1)[0]
            path = str(getattr(sys.modules.get(name), "__file__", "") or "")
            if (
                name in self.before
                or top in sys.stdlib_module_names
                or top in ("unsloth", "unsloth_zoo", "tests", "__main__", "__mp_main__")
                or os.path.basename(path).startswith(("test_", "conftest"))
            ):
                continue
            self.first.append(name)
        # Now, not only at the conftest or exit: some harnesses end their ranks with os._exit. Until a
        # conftest run is seen this says "no conftest", which only makes the next zygote stop short of it.
        self.report(final = False)

    def report(self, final = True):
        if self.reported or self.first is None:
            return
        self.reported = final
        event = {
            "target": self.key,
            "event": "observed",
            "first": self.first,
            # test modules and conftest-like helpers the rank's code imports are its own, never ahead
            "before_conftest": [
                n
                for n in self.before_conftest
                if not os.path.basename(
                    str(getattr(sys.modules.get(n), "__file__", "") or "")
                ).startswith(("test_", "conftest"))
            ],
            "conftest": self.conftest,
            "unsloth_first": self.unsloth_first,
        }
        path = self.ranks_file
        if os.path.isdir(path):
            path = os.path.join(path, f"{os.getpid()}.jsonl")  # one writer per file: no torn lines
        try:
            with open(path, "a", encoding = "utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except OSError:
            pass


# --------------------------------------------------------------------------------------------------


def summary(root: str) -> dict:
    counts = {"start": 0, "hit": 0, "respawn": 0}
    reasons = {}
    lowest = []  # (MemAvailable GB, test) at the events where the host had the least memory left
    for name in os.listdir(root) if os.path.isdir(root) else []:
        with open(os.path.join(root, name), encoding = "utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                kind = event.get("event")
                counts[kind] = counts.get(kind, 0) + 1
                if isinstance(event.get("mem_gb"), (int, float)) and event["mem_gb"] != float(
                    "inf"
                ):
                    lowest = sorted(lowest + [(round(event["mem_gb"], 2), event.get("test"))])[:5]
                if kind == "respawn":
                    reasons[event.get("reason")] = reasons.get(event.get("reason"), 0) + 1
                if kind == "spawn":
                    why = "spawned: " + str(event.get("why"))
                    reasons[why] = reasons.get(why, 0) + 1
    counts["no_conftest"] = counts["start"] - counts["hit"] - counts["respawn"]
    return {
        "counts": counts,
        "reasons": dict(sorted(reasons.items(), key = lambda kv: -kv[1])),
        "lowest_mem_available_gb": lowest,
    }


if os.environ.get(_SPEC_ENV) and __name__ == "rank_zygote":
    # Imported as the forkserver's preload.
    _spec = os.environ.pop(_SPEC_ENV)
    _preload(_spec)

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--summary":
        print(json.dumps(summary(sys.argv[2]), indent = 2))

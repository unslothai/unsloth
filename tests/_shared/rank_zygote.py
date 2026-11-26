# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Forks spawned test ranks from a process that has already run ``tests/conftest.py`` and ``import unsloth``.

Every gloo test spawns 2 to 8 ranks, and each spawned rank re-imports torch and unsloth: about 9 CPU seconds
of a rank's 11 on a CI runner. With ``UNSLOTH_TEST_RANK_ZYGOTE=1`` (set by distributed-cpu-ci.yml only) a
``multiprocessing`` spawn start in a test process is served instead by a per-process zygote, a stdlib
forkserver whose preload is this module. The zygote imports torch and pytest (the base), records the base
state, then runs the conftest the way the rank harnesses do (``spec_from_file_location`` + ``exec_module``)
under a recorder of every ``os.environ`` access, and forks each rank from that state.

A forked rank must end up as a spawned rank would have, so it starts with the parent's environment and
stdio, the modules the conftest imported hidden, and the base attributes they changed put back. Nothing is
reused until the rank itself executes ``tests/conftest.py`` (the import point). There the zygote's state
replaces the conftest run only when the result cannot differ:

- the rank has imported none of the hidden modules (unsloth, transformers, ...) on its own,
- every environment variable the zygote's import read has the same value in the rank, and
- the base attributes the import changes are still as the zygote found them.

Anything else, a plain ``import unsloth`` outside the conftest included, re-executes the rank as a fresh
spawned interpreter (same pid, same pipes), so it runs exactly as without this module, only later.
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

# Base: imported before the conftest runs and shared by every rank as is; test modules import these first.
# bitsandbytes too: the conftest imports it before anything else it does (before its CUDA spoof), and rank
# harnesses import it themselves before the conftest. The standard library modules the import pulls in join
# them (found by a throwaway import in a fork of the zygote), so a rank importing one of those is no reason
# to start again.
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
            f.write(
                json.dumps({"event": kind, "pid": os.getpid(), "t": time.time(), **fields}) + "\n"
            )
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
_turns = {}  # layout key -> ranks started on it, for the hash seed alternation
MAX_ZYGOTES_ENV = "UNSLOTH_TEST_RANK_ZYGOTE_MAX"


def install() -> bool:
    """Route this process's multiprocessing spawn starts through the zygote; False when it does not apply."""
    global _original_popen
    import multiprocessing

    if not enabled() or _IN_ZYGOTE or multiprocessing.parent_process() is not None:
        return False  # the pytest process only: not the zygote's own conftest run, not a rank's
    from multiprocessing import popen_spawn_posix

    if _original_popen is None:
        _original_popen = popen_spawn_posix.Popen
        popen_spawn_posix.Popen = _popen
    return True


_seen = {}  # target -> [hits, respawns because the rank imported a hidden module itself]
_offsets = {}  # ranks.jsonl path -> bytes read


def _learn():
    for zygote in list(_servers.values()) + list(_retired):
        path = os.path.join(zygote.dir, "ranks.jsonl")
        try:
            with open(path, encoding = "utf-8") as f:
                f.seek(_offsets.get(path, 0))
                lines = f.readlines()
                _offsets[path] = f.tell()
        except OSError:
            continue
        for line in lines:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            counts = _seen.setdefault(event.get("target"), [0, 0])
            if event.get("event") == "hit":
                counts[0] += 1
            elif str(event.get("reason", "")).startswith("imported "):
                counts[1] += 1


def _popen(process_obj):
    if not enabled():
        return _original_popen(process_obj)
    _learn()
    hits, imports = _seen.get(_target_name(process_obj), (0, 0))
    if imports >= 2 and hits == 0:
        # This function's ranks import a hidden module before the conftest every time, so they would only
        # start again from scratch: spawn them directly.
        return _original_popen(process_obj)
    zygote = _zygote_for(process_obj)
    if zygote is None or zygote.status().get("broken"):
        return _original_popen(process_obj)
    return _ZygotePopen(process_obj, zygote)


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


def _zygote_for(process_obj):
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
    env.update(guess)
    key = json.dumps(sorted(guess.items()))
    known = _servers.get(json.dumps([key, 0]))
    world = _world_arg(process_obj)
    if known is not None and world is not None and known.status().get("rank_inputs"):
        # This layout's import reads WORLD_SIZE where it matters (UNSLOTH_PARALLEL translates by it).
        env["WORLD_SIZE"] = str(world)
        key = json.dumps(sorted({**guess, "WORLD_SIZE": str(world)}.items()))
    # Spawned ranks each draw their own string hash seed, so a set iterated in a different order on two
    # ranks shows. A fork keeps its zygote's seed: consecutive ranks alternate between two zygotes with
    # different seeds, unless PYTHONHASHSEED fixes one seed for every process anyway.
    fixed = os.environ.get("PYTHONHASHSEED", "random") != "random"
    turn = _turns[key] = _turns.get(key, -1) + 1
    key = json.dumps([key, 0 if fixed else turn % 2])
    zygote = _servers.pop(key, None)
    if zygote is None:
        limit = max(1, int(os.environ.get(MAX_ZYGOTES_ENV, "4") or "4"))
        for old_key in list(_servers):
            if len(_servers) < limit:
                break
            if _servers[old_key].idle():
                old = _servers.pop(old_key)
                old.stop()
                _retired.append(old)
        if len(_servers) >= limit:
            return None  # every zygote still has a live rank: a plain spawn
        zygote = _Zygote(env)
    _servers[key] = zygote  # most recently used last
    return zygote


class _Zygote:
    """One forkserver whose preload imported unsloth under ``env``."""

    _count = 0

    def __init__(self, env):
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
                (z.status()["stdlib"] for z in _servers.values() if "stdlib" in z.status()), None
            ),
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

        def __init__(self, process_obj, zygote):
            self._zygote = zygote
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
                "target": _target_name(process_obj),
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
            stdlib = _stdlib_imported_by_the_conftest()
        late_recorder = _EnvRecorder()
        late_recorder.start()
        try:
            for name in ("bitsandbytes", *stdlib):
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
        base_inputs = {
            **late_inputs,
            **{k: v for k, (kind, v) in base_recorder.first.items() if kind == "read"},
        }
        # torch, numpy and pytest come first in a rank too (its test module imports them before running the
        # conftest); the late ones a rank imports during the conftest, after the variables it sets.
        written = sorted(
            set(late_inputs) & {k for k, (kind, _) in recorder.first.items() if kind == "write"}
        )
        if written:
            raise RuntimeError(
                f"bitsandbytes / stdlib imports read {written}, which the conftest writes"
            )
        _Z = {
            "spec": spec,
            "hidden": hidden,
            "changes": changes,
            "globals_before": globals_before,
            "globals_after": _globals_snapshot(),
            "base_env_inputs": base_inputs,
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
            "conftest": dict(vars(conftest)),
            "rank_neutral": torch.cuda.device_count() == 0,
        }
        unknown_scans = sorted(s for s in recorder.scans if s not in _PREFIX_SCANS)
        status.update(
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
    for key, value in _Z["base_env_inputs"].items():
        if os.environ.get(key) != value:
            _respawn(f"env {key} read by the base imports")
    _hide()

    stream = io.BytesIO(data)
    process.current_process()._inheriting = True
    try:
        preparation_data = pickle.load(stream)
        spawn.prepare(preparation_data)
        self = pickle.load(stream)
    finally:
        del process.current_process()._inheriting
    return self._bootstrap(parent_sentinel)


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
        if _rank["decided"] is None and _is_conftest(getattr(module, "__file__", None)):
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
    for key, value in _Z["base_env_inputs"].items():
        if env.get(key) != value:
            return f"env {key} read by the base imports"
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
            if dict(env) != _Z["spec"]["env"]:
                return f"environment scanned by {scan}"
            continue
        ours = {k: v for k, v in env.items() if k.startswith(prefix)}
        theirs = {k: v for k, v in _Z["spec"]["env"].items() if k.startswith(prefix)}
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
        {"data": data_fd, "sentinel": _rank["sentinel"], "inherited": inherited, "tracker": tracker}
    )
    code = (
        f"import sys; sys.path.insert(0, {_HERE!r}); import rank_zygote; "
        f"sys.path.remove({_HERE!r}); rank_zygote._respawned({args!r})"
    )
    exe = sys.executable
    os.execve(exe, [exe, *util._args_from_interpreter_flags(), "-c", code], _rank["env"])


def _respawned(args_json: str):
    """The re-executed rank: what spawn_main does, on the payload the zygote received."""
    from multiprocessing import forkserver, resource_tracker, spawn

    args = json.loads(args_json)
    forkserver._forkserver._inherited_fds = args["inherited"]
    resource_tracker._resource_tracker._fd = args["tracker"]
    sys.exit(spawn._main(args["data"], args["sentinel"]))


# --------------------------------------------------------------------------------------------------


def summary(root: str) -> dict:
    counts = {"start": 0, "hit": 0, "respawn": 0}
    reasons = {}
    for name in os.listdir(root) if os.path.isdir(root) else []:
        with open(os.path.join(root, name), encoding = "utf-8") as f:
            for line in f:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                kind = event.get("event")
                counts[kind] = counts.get(kind, 0) + 1
                if kind == "respawn":
                    reasons[event.get("reason")] = reasons.get(event.get("reason"), 0) + 1
    counts["no_conftest"] = counts["start"] - counts["hit"] - counts["respawn"]
    return {
        "counts": counts,
        "respawn_reasons": dict(sorted(reasons.items(), key = lambda kv: -kv[1])),
    }


if os.environ.get(_SPEC_ENV) and __name__ == "rank_zygote":
    # Imported as the forkserver's preload.
    _spec = os.environ.pop(_SPEC_ENV)
    _preload(_spec)

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--summary":
        print(json.dumps(summary(sys.argv[2]), indent = 2))

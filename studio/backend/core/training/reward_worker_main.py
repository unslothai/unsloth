# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runs inside the sandbox and scores batches for Python rewards. Standard library only.

The sandbox gives the process no stdin (MXC), so requests and replies are files in the workdir:
``req-<n>.json`` in, ``res-<n>.json`` out, each written to a temp name and renamed into place.
"""

import json
import math
import os
import sys
import time
import traceback

POLL_SECONDS = 0.002
# The host touches "alive" every few seconds; a host that died without saying stop goes quiet.
ALIVE_TIMEOUT_SECONDS = 60.0


def _retry(action, seconds = 10.0):
    # Windows: a virus scanner can hold a file it just saw appear for a few milliseconds.
    deadline = time.monotonic() + seconds
    while True:
        try:
            return action()
        except PermissionError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.005)


def put(path, payload):
    tmp = path + ".tmp"

    def write():
        with open(tmp, "w", encoding = "utf-8") as f:
            json.dump(payload, f)

    _retry(write)
    _retry(lambda: os.replace(tmp, path))


def take(path):
    def read():
        with open(path, encoding = "utf-8") as f:
            return json.load(f)

    data = _retry(read)
    _retry(lambda: os.remove(path))
    return data


def _error_line(exc):
    lines = traceback.format_exception_only(type(exc), exc)
    return (lines[-1] if lines else repr(exc)).strip()[:2000]


def _install_unsloth_helpers(path):
    """Notebook rewards do ``from unsloth import execute_with_time_limit`` and friends. Importing
    unsloth here would need a GPU the sandbox can't see, so load just those helpers from
    unsloth_zoo/rl_environments.py and serve them as ``unsloth``."""
    if not path or "unsloth" in sys.modules:
        return
    import importlib.util
    import types

    spec = importlib.util.spec_from_file_location("unsloth_rl_environments", path)
    helpers = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helpers
    spec.loader.exec_module(helpers)
    shim = types.ModuleType("unsloth")
    for name in getattr(helpers, "__all__", ()):
        setattr(shim, name, getattr(helpers, name))
    sys.modules["unsloth"] = shim


def _load(specs):
    funcs, errors = {}, {}
    for spec in specs:
        name = spec["name"]
        namespace = {"__name__": "reward_" + name.replace("-", "_"), "__builtins__": __builtins__}
        try:
            code = compile(spec["code"], "<reward " + name + ">", "exec")
            exec(code, namespace)
            fn = namespace.get(spec["entry"])
            if not callable(fn):
                raise NameError(spec["entry"] + " is not defined by the reward code")
            funcs[name] = fn
        except BaseException as exc:  # noqa: BLE001 - reported back, never kills the worker
            errors[name] = _error_line(exc)
    return funcs, errors


def _score(fn, request):
    out = fn(
        prompts = request.get("prompts"),
        completions = request.get("completions"),
        **(request.get("kwargs") or {}),
    )
    n = len(request.get("completions") or [])
    out = list(out) if out is not None else []
    if len(out) != n:
        raise ValueError("returned %d scores for %d completions" % (len(out), n))
    scores = []
    for value in out:
        if value is None:
            scores.append(None)
            continue
        value = float(value)
        scores.append(value if math.isfinite(value) else None)
    return scores


def main():
    workdir = os.getcwd()
    # Notebook rewards print a lot; a pipe nobody drains would block them.
    log = open(os.path.join(workdir, "worker.log"), "a", encoding = "utf-8", buffering = 1)
    sys.stdout = sys.stderr = log
    with open(os.path.join(workdir, "rewards.json"), encoding = "utf-8") as f:
        setup = json.load(f)
    specs = setup["specs"]
    try:
        _install_unsloth_helpers(setup.get("unsloth_helpers"))
    except Exception as exc:  # noqa: BLE001 - only rewards that use them will fail, by name
        print("Could not load the unsloth notebook helpers:", _error_line(exc))
    funcs, errors = _load(specs)
    put(os.path.join(workdir, "ready.json"), {"loaded": sorted(funcs), "errors": errors})
    seq = 0
    alive_path = os.path.join(workdir, "alive")
    next_check = 0.0
    while True:
        req_path = os.path.join(workdir, "req-%d.json" % seq)
        if not os.path.exists(req_path):
            if os.path.exists(os.path.join(workdir, "stop")):
                return 0
            now = time.monotonic()
            if now >= next_check:
                next_check = now + 1.0
                try:
                    quiet = time.time() - os.path.getmtime(alive_path)
                except OSError:
                    return 0
                if quiet > ALIVE_TIMEOUT_SECONDS:
                    return 0
            time.sleep(POLL_SECONDS)
            continue
        request = take(req_path)
        result = {"scores": {}, "errors": {}}
        for name in request.get("rewards") or []:
            fn = funcs.get(name)
            if fn is None:
                result["errors"][name] = errors.get(name, "reward is not loaded")
                continue
            try:
                result["scores"][name] = _score(fn, request)
            except BaseException as exc:  # noqa: BLE001 - one bad reward must not end the run's worker
                result["errors"][name] = _error_line(exc)
        put(os.path.join(workdir, "res-%d.json" % seq), result)
        seq += 1


if __name__ == "__main__":
    sys.exit(main())

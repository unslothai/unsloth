# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""tests/_shared/rank_zygote.py: a forked rank ends up as a spawned rank would.

Each case runs a driver with UNSLOTH_TEST_RANK_ZYGOTE=0 and =1 and compares the ranks' import-time state."""

import json
import multiprocessing
import os
import subprocess
import sys

import pytest
import torch  # noqa: F401  (as every gloo test module: imported before the rank function runs)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CONFTEST = os.path.join(REPO, "tests", "conftest.py")

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "fd passing / memfd: Linux"),
    # One worker runs `runs`: three at once ran a 16 GB runner out of memory.
    pytest.mark.xdist_group("rank_zygote"),
    # Two drivers importing unsloth take minutes on a busy runner, over the shard's 330 s.
    pytest.mark.timeout(900),
]


def _fingerprint():
    """The state a gloo rank harness builds on, after it ran the conftest and imported unsloth."""
    import warnings

    import torch
    import transformers

    import unsloth  # noqa: F401
    from unsloth import device_type

    keep = (
        "UNSLOTH_",
        "TORCHINDUCTOR_",
        "TRITON_",
        "ACCELERATE_",
        "HF_",
        "TOKENIZERS_",
        "OMP_",
        "CUDA_",
    )
    env = {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith(keep) and not k.startswith("UNSLOTH_TEST_RANK_ZYGOTE")
    }
    # Written from a set: its order follows the process's string hash seed, which differs per spawned rank too.
    env["UNSLOTH_IGNORED_TOKENIZER_NAMES"] = sorted(
        env.get("UNSLOTH_IGNORED_TOKENIZER_NAMES", "").split("\n")
    )
    return {
        "env": env,
        "device_type": device_type.DEVICE_TYPE,
        "trainer_init": f"{transformers.Trainer.__init__.__module__}.{transformers.Trainer.__init__.__qualname__}",
        "manual_seed": torch.manual_seed.__module__,
        "adamw_init_group": torch.optim.AdamW._init_group.__module__,
        "warnings": len(warnings.filters),
        "unsloth_file": os.path.realpath(unsloth.__file__),
        "parent": multiprocessing.parent_process() is not None,
        "inheriting_flag_seen": _INHERITING,
        "str_hash": hash("rank zygote"),  # compared across ranks, not across runs
    }


# What a test module of a gloo harness records at import in a rank (they set CUDA_VISIBLE_DEVICES from it).
_INHERITING = bool(getattr(multiprocessing.current_process(), "_inheriting", False))


def _rank_via_conftest(
    rank,
    world,
    env,
    out,
    import_first = None,
):
    """The harness pattern: rank variables and the test's env, then the conftest by path, then unsloth."""
    os.environ.update(RANK = str(rank), WORLD_SIZE = str(world), LOCAL_RANK = str(rank))
    os.environ.update(env)
    if import_first:
        __import__(import_first)
    import importlib.util

    spec = importlib.util.spec_from_file_location("_tp_conftest", CONFTEST)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    with open(out, "w", encoding = "utf-8") as f:
        json.dump(_fingerprint(), f, sort_keys = True)


def _rank_exit(code, queue):
    queue.put(("alive", os.getpid()))
    queue.close()
    queue.join_thread()  # flushed before os._exit skips the feeder thread
    if code == "raise":
        raise RuntimeError("raised in the rank")
    os._exit(code)


def _drive(tmp, cases):
    """Runs in a fresh interpreter (the pytest process of a shard): spawns the ranks of each case."""
    sys.path.insert(0, os.path.join(HERE, "_shared"))
    import importlib.util

    spec = importlib.util.spec_from_file_location("_drive_conftest", CONFTEST)
    spec.loader.exec_module(
        importlib.util.module_from_spec(spec)
    )  # installs the zygote when enabled
    ctx = multiprocessing.get_context("spawn")
    report = {}
    for name, (env, import_first) in cases.items():
        outs = [os.path.join(tmp, f"{name}_{r}.json") for r in range(2)]
        procs = [
            ctx.Process(target = _rank_via_conftest, args = (r, 2, env, outs[r], import_first))
            for r in range(2)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(600)
        report[name] = {
            "codes": [p.exitcode for p in procs],
            "ranks": [
                json.load(open(o, encoding = "utf-8")) if os.path.exists(o) else None for o in outs
            ],
        }
    queue = ctx.Queue()
    codes = {}
    for code in (0, 7, "raise"):
        p = ctx.Process(target = _rank_exit, args = (code, queue))
        p.start()
        p.join(600)
        codes[str(code)] = [p.exitcode, queue.get(timeout = 60)[0]]
    report["exit_codes"] = codes
    with open(os.path.join(tmp, "report.json"), "w", encoding = "utf-8") as f:
        json.dump(report, f, sort_keys = True)


# The first ranks of a kind run as plain spawns under an observer; later ones of the same kind fork.
CASES = {
    "plain": ({}, None),
    "tp_layout": (
        {"UNSLOTH_TP_SIZE": "2"},
        None,
    ),  # same function: forked from a zygote keyed by the layout
    "plain_again": ({}, None),
    "imports_first": ({}, "transformers"),  # a new kind (its string argument): observed
    # The rank imports transformers itself before the conftest, so its zygote stops short of the conftest.
    "imports_first_again": ({}, "transformers"),
}


def _run(tmp_path, zygote):
    tmp = tmp_path / f"zygote{zygote}"
    tmp.mkdir()
    stats = tmp / "stats"
    env = dict(os.environ)
    env.update(
        UNSLOTH_TEST_RANK_ZYGOTE = str(zygote),
        UNSLOTH_TEST_RANK_ZYGOTE_STATS = str(stats),
        UNSLOTH_TEST_RANK_ZYGOTE_HOT = "0",  # a zygote from the first rank of a kind on, to keep this short
        CUDA_VISIBLE_DEVICES = "",
        PYTHONPATH = os.pathsep.join(p for p in (HERE, REPO, env.get("PYTHONPATH")) if p),
        UNSLOTH_TEST_RANK_ZYGOTE_MAX = "4",  # a fixed cap, whatever the shard's worker count
    )
    for name in ("PYTEST_XDIST_TESTRUNUID", "PYTEST_XDIST_WORKER", "PYTEST_XDIST_WORKER_COUNT"):
        env.pop(name, None)  # its own learning, not the observations of the shard around it
    code = f"import test_rank_zygote_gloo as t; t._drive({str(tmp)!r}, t.CASES)"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env = env,
        cwd = REPO,
        stdin = subprocess.DEVNULL,
        start_new_session = True,
        capture_output = True,
        text = True,
        timeout = 1500,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    report = json.loads((tmp / "report.json").read_text(encoding = "utf-8"))
    sys.path.insert(0, os.path.join(HERE, "_shared"))
    import rank_zygote

    return report, rank_zygote.summary(str(stats))


@pytest.fixture(scope = "module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rank_zygote")
    spawned, _ = _run(tmp, 0)
    forked, summary = _run(tmp, 1)
    return spawned, forked, summary


def test_a_forked_rank_reports_what_a_spawned_rank_does(runs):
    spawned, forked, _ = runs
    for name in CASES:
        assert spawned[name]["codes"] == [0, 0], spawned[name]
        assert forked[name]["codes"] == [0, 0], forked[name]
        for rank in range(2):
            got, want = (dict(r[name]["ranks"][rank]) for r in (forked, spawned))
            got.pop("str_hash"), want.pop("str_hash")
            assert got == want, (name, rank)


def test_two_ranks_hash_strings_differently(runs):
    """Spawned ranks draw their own hash seeds, so set order can differ between ranks; forked ones too."""
    if os.environ.get("PYTHONHASHSEED", "random") != "random":
        pytest.skip("PYTHONHASHSEED fixes one seed for every process")
    for report in runs[:2]:
        for name in ("plain", "tp_layout"):
            ranks = report[name]["ranks"]
            assert ranks[0]["str_hash"] != ranks[1]["str_hash"], name


def test_the_env_layout_reaches_the_import(runs):
    """UNSLOTH_TP_SIZE is read while unsloth imports (it installs the tensor parallel Trainer hook), so the
    layout case must differ from the plain one in both runs, not only where the zygote was skipped."""
    spawned, forked, _ = runs
    for report in (spawned, forked):
        assert (
            report["tp_layout"]["ranks"][0]["trainer_init"]
            != report["plain"]["ranks"][0]["trainer_init"]
        )


def test_the_zygote_path_is_taken(runs):
    _, _, summary = runs
    counts, reasons = summary["counts"], summary["reasons"]
    # imports_first's 2 ranks import transformers before the conftest, so they respawn.
    assert counts["hit"] >= 4, summary
    assert counts["start"] >= 8, summary
    assert counts["respawn"] <= 2, summary
    assert set(reasons) <= {
        "spawned: observed",
        "imported transformers before the conftest",
    }, summary


def test_exit_codes_exceptions_and_queues_come_back(runs):
    spawned, forked, _ = runs
    assert forked["exit_codes"] == spawned["exit_codes"]
    assert forked["exit_codes"] == {"0": [0, "alive"], "7": [7, "alive"], "raise": [1, "alive"]}

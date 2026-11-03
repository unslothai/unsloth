# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A spawned child imports the test module before `parent_process()` is set, so the gloo-test CPU
guard must also check `current_process()._inheriting` or the "CPU" ranks run on the GPU."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["_UNSLOTH_TEST_GUARD_FIRED"] = "1"

import glob  # noqa: E402

import pytest  # noqa: E402

GUARDED = "parent_process() is not None"


def _child(out):
    out.put((os.environ.get("_UNSLOTH_TEST_GUARD_FIRED"), os.environ.get("CUDA_VISIBLE_DEVICES")))


def test_guard_fires_in_a_spawned_child():
    ctx = multiprocessing.get_context("spawn")
    out = ctx.Queue()
    p = ctx.Process(target = _child, args = (out,))
    p.start()
    fired, mask = out.get(timeout = 120)
    p.join(timeout = 60)
    assert fired == "1" and mask == "", (fired, mask)
    assert os.environ.get("_UNSLOTH_TEST_GUARD_FIRED") is None


def test_every_gloo_guard_also_checks_inheriting():
    here = os.path.dirname(os.path.abspath(__file__))
    stale = []
    for path in sorted(glob.glob(os.path.join(here, "*.py"))):
        text = open(path, encoding = "utf-8").read()
        if GUARDED in text and "_inheriting" not in text:
            stale.append(os.path.basename(path))
    assert not stale, f"guard without the import-time check: {stale}"

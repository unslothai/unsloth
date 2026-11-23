# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Whether a forked child of this process can still run autograd.

Once a process has run a backward on a torch build with a GPU, autograd's device threads exist and
every forked child refuses backward ("Unable to handle autograd's threading in combination with
fork-based multiprocessing"). On a CPU-only host that never happens, so fork-based gloo tests only
break on real GPU hosts, and only after an earlier test in the same worker ran a backward."""

from __future__ import annotations

import os


def _probe() -> None:
    import torch
    try:
        torch.ones(1, requires_grad = True).sum().backward()
    except Exception:
        os._exit(1)
    os._exit(0)


def fork_can_backward(timeout: float = 120.0) -> bool:
    import multiprocessing

    if "fork" not in multiprocessing.get_all_start_methods():
        return False
    child = multiprocessing.get_context("fork").Process(target = _probe)
    child.start()
    child.join(timeout)
    if child.is_alive():
        child.kill()
        child.join()
        return False
    return child.exitcode == 0

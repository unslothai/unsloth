# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""accelerate's init_empty_weights made per-thread (utils.meta_init). Stock accelerate patches
torch.nn.Module.register_parameter process-wide, so overlapping blocks in two threads left it patched for good and
leaked meta parameters into modules other threads built meanwhile."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time

import pytest

pytest.importorskip("accelerate")

from utils import meta_init


@pytest.fixture(autouse = True)
def _installed():
    assert meta_init.install()


def test_overlapping_blocks_in_two_threads_leave_nothing_behind():
    import torch.nn as nn
    from accelerate import init_empty_weights

    a_in, b_in = threading.Event(), threading.Event()
    built = {}

    def a():
        with init_empty_weights(include_buffers = False):
            a_in.set()
            b_in.wait(5)
            time.sleep(0.05)
            built["a"] = nn.Linear(4, 4)

    def b():
        a_in.wait(5)
        with init_empty_weights(include_buffers = False):
            b_in.set()
            time.sleep(0.3)
            built["b"] = nn.Linear(4, 4)

    threads = [threading.Thread(target = a), threading.Thread(target = b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert built["a"].weight.is_meta and built["b"].weight.is_meta
    assert not nn.Linear(4, 4).weight.is_meta


def test_a_block_does_not_leak_into_another_thread():
    import torch.nn as nn
    from accelerate import init_empty_weights

    inside, done = threading.Event(), threading.Event()

    def holder():
        with init_empty_weights(include_buffers = False):
            inside.set()
            done.wait(5)

    t = threading.Thread(target = holder)
    t.start()
    inside.wait(5)
    try:
        assert not nn.Linear(4, 4).weight.is_meta
    finally:
        done.set()
        t.join()


def test_nesting_and_buffers_behave_like_accelerate():
    import torch
    import torch.nn as nn
    from accelerate import init_empty_weights, init_on_device

    with init_empty_weights(include_buffers = False):
        bn = nn.BatchNorm1d(4)
        with init_on_device(torch.device("cpu"), include_buffers = False):
            inner = nn.Linear(2, 2)
        outer = nn.Linear(2, 2)
    assert bn.weight.is_meta and not bn.running_mean.is_meta
    assert not inner.weight.is_meta and outer.weight.is_meta
    with init_empty_weights(include_buffers = True):
        bn2 = nn.BatchNorm1d(4)
    assert bn2.weight.is_meta and bn2.running_mean.is_meta


def test_install_on_import_patches_accelerate_when_it_is_first_imported(tmp_path):
    import pathlib

    backend = str(pathlib.Path(meta_init.__file__).resolve().parents[1])
    code = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {backend!r})
        from utils import meta_init
        meta_init.install_on_import()
        assert "accelerate" not in sys.modules and "torch" not in sys.modules
        from accelerate import init_empty_weights, big_modeling
        assert meta_init._installed, "not installed on import"
        import accelerate
        assert accelerate.init_on_device is big_modeling.init_on_device
        print("ok")
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output = True, text = True, timeout = 120)
    assert proc.returncode == 0 and "ok" in proc.stdout, proc.stderr[-2000:]

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Training a model split across devices fails, and it should say so.

`accelerate` shards across every visible card by default, so a user with two
GPUs and no ``device_map`` gets the split without asking for it. The training
loop then announces ``Num GPUs used = 2`` and dies at the first embedding
lookup:

    RuntimeError: Expected all tensors to be on the same device, but got index
    is on cuda:0, different from other tensors on cuda:1
    (when checking argument in method wrapper_CUDA__index_select)

which reads as a tensor bug rather than a placement one. Reproduced on 2x Tesla
T4 with ``unsloth/Qwen3-0.6B`` in 4bit: 232849408 parameters on cuda:0 and
155582464 on cuda:1, dying at step 0.

Aligning that one call site would only move the error -- ``llama.py`` allocates
working buffers on device 0 in several other places -- so the loop refuses,
the same answer it already gives for TPUs a few lines below.

The guard is generated source spliced into ``_inner_training_loop``, so these
rules extract the literal and RUN it. Asserting that the text is present would
pass on a guard that never fires.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

LLAMA = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "llama.py"
SRC = LLAMA.read_text(encoding = "utf-8")


class _Param:
    def __init__(self, device):
        self.device = _Device(device)


class _Device:
    def __init__(self, spec):
        self.type = spec.split(":")[0]
        self._spec = spec

    def __str__(self):
        return self._spec


class _Model:
    def __init__(self, devices):
        self._devices = devices

    def parameters(self):
        return [_Param(d) for d in self._devices]


def _guard_source() -> str:
    """The literal the training loop splices in, re-indented as it re-indents it.

    EXEC'd out of the file rather than read as text, because the literal is
    escaped twice: `\\n` in the file is `\\n` in the string and a newline only
    once the generated source is compiled. Reading it with a regex gives the
    outer layer and runs none of it.
    """
    start = SRC.index('        multi_gpu_guard = """')
    end = SRC.index("multi_gpu_guard = multi_gpu_guard.split", start)
    scope: dict = {}
    exec("if True:\n" + SRC[start:end], scope)
    body = scope["multi_gpu_guard"]
    # The same re-indent the loop applies, with no leading whitespace since
    # this runs at module level rather than inside a method.
    lines = body.split("\n")
    body = "\n".join([lines[0]] + [x[8:] for x in lines[1:]])
    # The loop appends `debug_info =` so the banner assignment continues from
    # it; that trailing fragment is not runnable on its own.
    return body.rsplit("debug_info =", 1)[0]


def _run(devices, env = None):
    scope = {"model": _Model(devices), "os": os}
    if env is not None:
        os.environ["UNSLOTH_ALLOW_MULTI_GPU"] = env
    else:
        os.environ.pop("UNSLOTH_ALLOW_MULTI_GPU", None)
    try:
        exec(_guard_source(), scope)
    finally:
        os.environ.pop("UNSLOTH_ALLOW_MULTI_GPU", None)


def test_a_split_model_is_refused_with_the_devices_named():
    with pytest.raises(RuntimeError) as excinfo:
        _run(["cuda:0", "cuda:0", "cuda:1"])
    message = str(excinfo.value)
    # The devices, so the reader can see WHICH cards without instrumenting.
    assert "cuda:0" in message and "cuda:1" in message
    # And the two ways out, spelled out. An error that says only "unsupported"
    # leaves the user to find this out by searching issues.
    assert "CUDA_VISIBLE_DEVICES=0" in message
    assert 'device_map = {"": 0}' in message


def test_one_device_is_untouched():
    """The overwhelmingly common case, and it must cost nothing."""
    _run(["cuda:0", "cuda:0", "cuda:0"])
    _run(["cuda:1", "cuda:1"])


def test_DDP_is_not_caught_by_this():
    """Each DDP rank holds the WHOLE model on its own single device, so the set
    has one element per process. That is why the check reads the parameter
    devices rather than `torch.cuda.device_count()` or `args.world_size`, both
    of which are greater than one under DDP and would break it."""
    _run(["cuda:3", "cuda:3", "cuda:3"])


def test_cpu_parameters_do_not_count_as_a_second_device():
    """`offload_embedding` deliberately puts the input embedding on the CPU and
    installs hooks to move it, which is a supported configuration. Counting the
    CPU as a device would refuse it."""
    _run(["cpu", "cuda:0", "cuda:0"])


def test_the_escape_hatch_works_and_is_opt_in():
    with pytest.raises(RuntimeError):
        _run(["cuda:0", "cuda:1"], env = "0")
    _run(["cuda:0", "cuda:1"], env = "1")


def test_the_guard_runs_BEFORE_the_banner_announces_a_device_count():
    """Order matters for what the user sees. Announcing `Num GPUs used = 2` and
    then refusing reads as a contradiction; refusing first reads as an answer.
    """
    guard_at = SRC.index("multi_gpu_guard = ")
    splice = SRC.index(
        'inner_training_loop.replace(\n            "debug_info =", multi_gpu_guard, 1,\n        )'
    )
    banner_splice = SRC.index("inner_training_loop.replace(original_debug, debug_info)")
    assert guard_at < banner_splice
    # The guard is spliced in AFTER the banner is put in place, which is what
    # puts it textually ahead of the banner: it replaces the first
    # `debug_info =`, and that is the banner's own assignment.
    assert banner_splice < splice


def test_it_is_a_refusal_and_not_a_warning():
    """A warning here is worse than nothing: the run continues and dies anyway,
    with the warning scrolled off above a traceback about index_select."""
    body = _guard_source()
    assert "raise RuntimeError(" in body
    assert "warn" not in body.lower()

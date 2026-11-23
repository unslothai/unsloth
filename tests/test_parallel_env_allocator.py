# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""torchrun ranks > 0 bind their CUDA device at `import unsloth`, which starts CUDA before unsloth_zoo exports
its allocator default (expandable_segments). Those ranks then ran without it while rank 0 had it: on 2x T4,
Qwen3.8-27B 4-bit FSDP2 left rank 1 with 511 allocator segments and 2.2 GiB more reserved than rank 0."""

import importlib.util
import os
import subprocess
import sys
import textwrap
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parallel_env():
    spec = importlib.util.spec_from_file_location(
        "_unsloth_parallel_env_under_test", os.path.join(REPO, "unsloth", "parallel_env.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_torch(
    version,
    initialized,
    calls,
    hip = None,
):
    memory = types.SimpleNamespace(_set_allocator_settings = calls.append)
    cuda = types.SimpleNamespace(is_initialized = lambda: initialized, memory = memory)
    return types.SimpleNamespace(
        __version__ = version, version = types.SimpleNamespace(hip = hip), cuda = cuda
    )


@pytest.mark.parametrize(
    "version, before, after, expect",
    [
        (
            "2.14.1",
            {},
            {"PYTORCH_ALLOC_CONF": "expandable_segments:True"},
            "expandable_segments:True",
        ),
        (
            "2.9.1",
            {},
            {
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,roundup_power2_divisions:[>:32]",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            },
            "expandable_segments:True,roundup_power2_divisions:[>:32]",
        ),
        # UNSLOTH_VLLM_STANDBY: zoo strips expandable_segments the user had set; turn it off here too.
        (
            "2.14.1",
            {"PYTORCH_ALLOC_CONF": "expandable_segments:True"},
            {"PYTORCH_ALLOC_CONF": ""},
            "expandable_segments:False",
        ),
        (
            "2.14.1",
            {"PYTORCH_ALLOC_CONF": "x"},
            {"PYTORCH_ALLOC_CONF": "x"},
            None,
        ),  # zoo changed nothing
    ],
)
def test_late_allocator_env_is_applied_once_cuda_started(
    monkeypatch, version, before, after, expect
):
    env = _parallel_env()
    calls = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(version, True, calls))
    assert env.apply_late_allocator_env(before, environ = after) == expect
    assert calls == ([expect] if expect else [])


def test_torch_reads_the_environment_itself_before_cuda_starts(monkeypatch):
    env = _parallel_env()
    calls = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("2.14.1", False, calls))
    after = {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    assert env.apply_late_allocator_env({}, environ = after) is None
    assert calls == []


@pytest.mark.parametrize(
    "version, hip, before, after, expect",
    [
        # torch 2.10+ reads PYTORCH_CUDA_ALLOC_CONF before PYTORCH_ALLOC_CONF (c10/core/AllocatorConfig.cpp): a
        # rank > 0 started CUDA with the legacy expandable_segments:True, zoo then dropped the legacy name.
        (
            "2.14.1",
            None,
            {
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "PYTORCH_ALLOC_CONF": "max_split_size_mb:512",
            },
            {"PYTORCH_ALLOC_CONF": "max_split_size_mb:512"},
            "max_split_size_mb:512,expandable_segments:False",
        ),
        # ROCm before 2.10 reads PYTORCH_CUDA_ALLOC_CONF first, PYTORCH_HIP_ALLOC_CONF only when it is unset.
        (
            "2.9.1",
            "6.4",
            {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            {},
            "expandable_segments:False",
        ),
    ],
)
def test_late_allocator_env_reads_variables_in_torch_order(
    monkeypatch, version, hip, before, after, expect
):
    env = _parallel_env()
    calls = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(version, True, calls, hip = hip))
    assert env.apply_late_allocator_env(before, environ = after) == expect
    assert calls == [expect]


def test_deprecated_setter_neither_warns_nor_is_skipped(monkeypatch):
    """torch 2.10+ marks torch.cuda.memory._set_allocator_settings deprecated (FutureWarning on every call): every
    rank > 0 printed it at import, and with warnings as errors the settings were silently not applied."""
    import warnings

    env = _parallel_env()
    calls = []

    def deprecated_setter(value):
        warnings.warn(
            "torch.cuda._set_allocator_settings is deprecated. Use "
            "torch._C._accelerator_setAllocatorSettings instead.",
            FutureWarning,
            stacklevel = 2,
        )
        calls.append(value)

    fake = _fake_torch("2.14.1", True, calls)
    fake.cuda.memory._set_allocator_settings = deprecated_setter
    monkeypatch.setitem(sys.modules, "torch", fake)
    after = {"PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    with warnings.catch_warnings(record = True) as seen:
        warnings.simplefilter("error")
        got = env.apply_late_allocator_env({}, environ = after)
    assert got == "expandable_segments:True"
    assert calls == ["expandable_segments:True"]
    assert not seen


def test_cuda_started_reads_torch_without_importing_it(monkeypatch):
    env = _parallel_env()
    monkeypatch.delitem(sys.modules, "torch", raising = False)
    assert env.cuda_started() is False
    assert "torch" not in sys.modules
    for initialized in (False, True):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch("2.14.1", initialized, []))
        assert env.cuda_started() is initialized


_REAL_SETTER = textwrap.dedent(
    """
    import importlib.util, json, os, sys, warnings
    import torch
    spec = importlib.util.spec_from_file_location("pe", os.path.join(sys.argv[1], "unsloth", "parallel_env.py"))
    pe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pe)
    torch.cuda.is_initialized = lambda: True
    with warnings.catch_warnings(record = True) as seen:
        warnings.simplefilter("always")
        pe.apply_late_allocator_env({}, environ = {"PYTORCH_ALLOC_CONF": "expandable_segments:False",
                                                   "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"})
    print("RESULT", json.dumps([str(w.message) for w in seen if issubclass(w.category, FutureWarning)]))
    """
)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason = "needs torch")
def test_real_torch_setter_emits_no_future_warning():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "", HIP_VISIBLE_DEVICES = "")
    out = subprocess.run(
        [sys.executable, "-c", _REAL_SETTER, REPO],
        env = env,
        capture_output = True,
        text = True,
        timeout = 300,
    )
    lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    assert lines, out.stdout[-3000:] + out.stderr[-3000:]
    import json

    assert json.loads(lines[-1][len("RESULT ") :]) == []


_RANK1 = textwrap.dedent(
    """
    import json, os, torch
    # What bind_local_cuda_device does on a torchrun rank > 0 (torch.cuda.set_device(LOCAL_RANK)): CUDA starts
    # before unsloth_zoo is imported. This host has one GPU, so the device is 0.
    torch.cuda.set_device(0)
    import unsloth  # noqa: F401
    x = torch.empty(64 << 20, dtype = torch.uint8, device = "cuda")
    segs = torch.cuda.memory._snapshot()["segments"]
    print("RESULT", json.dumps({"expandable": any(s.get("is_expandable") for s in segs),
                                "env": os.environ.get("PYTORCH_ALLOC_CONF") or os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}))
    """
)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or not __import__("torch").cuda.is_available()
    or os.name == "nt",
    reason = "needs a CUDA device (expandable segments: Linux)",
)
def test_cuda_started_before_zoo_still_gets_expandable_segments():
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (
            "PYTORCH_ALLOC_CONF",
            "PYTORCH_CUDA_ALLOC_CONF",
            "PYTORCH_HIP_ALLOC_CONF",
            "UNSLOTH_VLLM_STANDBY",
        )
    }
    env["PYTHONPATH"] = REPO
    out = subprocess.run(
        [sys.executable, "-c", _RANK1],
        env = env,
        capture_output = True,
        text = True,
        timeout = 600,
        cwd = REPO,
    )
    lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    assert lines, out.stdout[-3000:] + out.stderr[-3000:]
    import json

    got = json.loads(lines[-1][len("RESULT ") :])
    if "expandable_segments:True" not in (got["env"] or ""):
        pytest.skip(f"unsloth_zoo keeps expandable_segments off on this host ({got['env']!r})")
    assert got["expandable"], got


_SINGLE = textwrap.dedent(
    """
    import json, torch
    calls = []
    original = torch.cuda.memory._set_allocator_settings
    torch.cuda.memory._set_allocator_settings = lambda value: (calls.append(value), original(value))[1]
    import unsloth  # noqa: F401
    print("RESULT", json.dumps({"started": torch.cuda.is_initialized(), "calls": calls}))
    """
)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or not __import__("torch").cuda.is_available(),
    reason = "needs a CUDA device",
)
def test_single_process_import_leaves_the_allocator_alone():
    """No torchrun, CUDA not started before `import unsloth`: zoo's own import starts CUDA after exporting its
    settings, so torch already read them and nothing is applied late (single-GPU runs stay as on main)."""
    # Unset like a fresh shell: zoo exports its settings into this process at import (another test may have
    # imported it), and a child inheriting them would see nothing to change.
    drop = ("LOCAL_RANK", "RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "UNSLOTH_VLLM_STANDBY") + (
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_CUDA_ALLOC_CONF",
        "PYTORCH_HIP_ALLOC_CONF",
    )
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env["PYTHONPATH"] = REPO
    out = subprocess.run(
        [sys.executable, "-c", _SINGLE],
        env = env,
        capture_output = True,
        text = True,
        timeout = 600,
        cwd = REPO,
    )
    lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
    assert lines, out.stdout[-3000:] + out.stderr[-3000:]
    import json

    got = json.loads(lines[-1][len("RESULT ") :])
    assert got["started"], got
    assert got["calls"] == [], got

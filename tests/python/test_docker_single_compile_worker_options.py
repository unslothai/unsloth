# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Regression guard for the pinned-device single-compile-worker block.

`docker --gpus '"device=N"'` sets NVIDIA_VISIBLE_DEVICES but not
CUDA_VISIBLE_DEVICES, so Inductor's compile workers cannot enumerate the
cgroup-pinned GPU and die with "Could not find an active GPU backend".
unsloth/_gpu_init.py therefore forces one in-process compile thread.

The trap this pins: the block runs AFTER `import unsloth_zoo`, and that import
chain (unsloth_zoo/__init__.py -> .temporary_patches -> .gpt_oss -> .common)
has already built its module-level Inductor options dicts from the original
4-32 thread count. Those dicts are plain snapshots handed to torch.compile as
`options`, and Inductor applies `options` as a config patch that outranks both
TORCHINDUCTOR_COMPILE_THREADS and torch._inductor.config.compile_threads. So
replacing determine_compile_threads alone leaves the guard ineffective for
every already-decorated compile site (rl_replacements, loss_utils,
cross_entropy_loss, temporary_patches.utils, the gpt_oss fused MoE paths).

Static + in-process: no docker, no GPU, no real torch compile. The guard block
is extracted from the shipped source and executed against a synthetic
unsloth_zoo module graph, so the test exercises the real code rather than a
copy of it.
"""

from __future__ import annotations

import functools
import importlib
import os
import re
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GPU_INIT = REPO_ROOT / "unsloth" / "_gpu_init.py"

SENTINEL = "UNSLOTH_FORCE_SINGLE_COMPILE_WORKER"
ORIGINAL_THREADS = 32  # what determine_compile_threads() returns on a 32+ core host


@pytest.fixture(scope="module")
def gpu_init_source() -> str:
    assert GPU_INIT.is_file(), f"missing {GPU_INIT}"
    return GPU_INIT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def guard_block(gpu_init_source: str) -> str:
    """The `if <sentinel> == "1":` block, verbatim from the shipped file."""
    start = re.search(
        r'^if os\.environ\.get\("%s", "0"\) == "1":$' % SENTINEL,
        gpu_init_source,
        re.MULTILINE,
    )
    assert start, "the single-compile-worker guard block disappeared"
    lines = gpu_init_source[start.start() :].splitlines()
    block = [lines[0]]
    for line in lines[1:]:
        if line and not line.startswith((" ", "\t")):
            break
        block.append(line)
    return "\n".join(block) + "\n"


def test_guard_runs_after_unsloth_zoo_is_imported(gpu_init_source: str):
    # The premise of the whole test: the zoo (and therefore its options dicts)
    # is already live by the time the block runs, so rewriting the dicts is the
    # only thing that can reach them.
    zoo_import = re.search(r"^    import unsloth_zoo$", gpu_init_source, re.MULTILINE)
    guard = re.search(
        r'^if os\.environ\.get\("%s", "0"\) == "1":$' % SENTINEL,
        gpu_init_source,
        re.MULTILINE,
    )
    assert zoo_import and guard
    assert zoo_import.start() < guard.start()


def _install_fake_zoo(monkeypatch) -> dict:
    """A stand-in for the zoo's post-import state on a 32-core host.

    Mirrors the real shapes: one shared dict re-exported by several modules
    (common -> loss_utils / rl_replacements / temporary_patches.utils) plus
    separate per-model dicts (gpt_oss's fused variants), and the
    functools.partial that closes over the shared dict.
    """
    shared = {"epilogue_fusion": True, "compile_threads": ORIGINAL_THREADS}
    fused = {"triton.cudagraphs": True, "compile_threads": ORIGINAL_THREADS}
    no_combo = {"combo_kernels": False, "compile_threads": ORIGINAL_THREADS}
    unrelated = {"compile_threads": ORIGINAL_THREADS}  # in a non-zoo module

    def _fake_determine_compile_threads():
        return ORIGINAL_THREADS

    zoo = types.ModuleType("unsloth_zoo")
    patches = types.ModuleType("unsloth_zoo.temporary_patches")
    common = types.ModuleType("unsloth_zoo.temporary_patches.common")
    gpt_oss = types.ModuleType("unsloth_zoo.temporary_patches.gpt_oss")
    loss_utils = types.ModuleType("unsloth_zoo.loss_utils")
    outsider = types.ModuleType("some_other_package")

    common.determine_compile_threads = _fake_determine_compile_threads
    common.torch_compile_options = shared
    common.torch_compile = functools.partial(lambda *a, **k: None, options=shared)
    gpt_oss.fused_torch_compile_options = fused
    gpt_oss.no_combo_fused_torch_compile_options = no_combo
    loss_utils.torch_compile_options = shared  # same object, re-exported
    patches.torch_compile_options = shared
    outsider.torch_compile_options = unrelated

    for module in (zoo, patches, common, gpt_oss, loss_utils, outsider):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    return {
        "shared": shared,
        "fused": fused,
        "no_combo": no_combo,
        "unrelated": unrelated,
        "common": common,
    }


def _run_guard(guard_block: str, monkeypatch) -> dict:
    state = _install_fake_zoo(monkeypatch)
    monkeypatch.setenv(SENTINEL, "1")
    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)

    fake_torch = types.SimpleNamespace(
        _inductor=types.SimpleNamespace(
            config=types.SimpleNamespace(compile_threads=ORIGINAL_THREADS)
        )
    )
    namespace = {
        "os": os,
        "sys": sys,
        "importlib": importlib,
        "torch": fake_torch,
        "__builtins__": __builtins__,
    }
    exec(compile(guard_block, str(GPU_INIT), "exec"), namespace)
    state["torch"] = fake_torch
    return state


def test_cached_options_dicts_are_rewritten(guard_block: str, monkeypatch):
    state = _run_guard(guard_block, monkeypatch)

    assert state["shared"]["compile_threads"] == 1, (
        "common.torch_compile_options is a snapshot built during `import "
        "unsloth_zoo`; left at %d it is passed to torch.compile as `options` and "
        "spawns the compile workers the guard exists to prevent" % ORIGINAL_THREADS
    )
    assert state["fused"]["compile_threads"] == 1
    assert state["no_combo"]["compile_threads"] == 1
    # The partial holds the same dict object, so the in-place rewrite reaches it.
    assert state["common"].torch_compile.keywords["options"]["compile_threads"] == 1


def test_guard_still_sets_env_config_and_function(guard_block: str, monkeypatch):
    state = _run_guard(guard_block, monkeypatch)

    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
    assert state["torch"]._inductor.config.compile_threads == 1
    # Dicts built after this point (compiler.py, gpt_oss's runtime rebuilds) go
    # through determine_compile_threads, which must now report 1.
    assert state["common"].determine_compile_threads() == 1


def test_guard_leaves_non_zoo_modules_alone(guard_block: str, monkeypatch):
    state = _run_guard(guard_block, monkeypatch)
    assert state["unrelated"]["compile_threads"] == ORIGINAL_THREADS


def test_guard_is_a_no_op_when_the_user_opted_out(guard_block: str, monkeypatch):
    state = _install_fake_zoo(monkeypatch)
    monkeypatch.delenv(SENTINEL, raising=False)
    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)

    fake_torch = types.SimpleNamespace(
        _inductor=types.SimpleNamespace(
            config=types.SimpleNamespace(compile_threads=ORIGINAL_THREADS)
        )
    )
    exec(
        compile(guard_block, str(GPU_INIT), "exec"),
        {
            "os": os,
            "sys": sys,
            "importlib": importlib,
            "torch": fake_torch,
            "__builtins__": __builtins__,
        },
    )

    assert state["shared"]["compile_threads"] == ORIGINAL_THREADS
    assert fake_torch._inductor.config.compile_threads == ORIGINAL_THREADS
    assert "TORCHINDUCTOR_COMPILE_THREADS" not in os.environ

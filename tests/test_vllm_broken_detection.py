# Unsloth - 2x faster, 60% less VRAM LLM training and finetuning
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

"""Regression test for issue #6590: ``fast_inference=True`` crashed with
``ImportError: libcudart.so.13`` because modern vLLM loads its compiled
extensions lazily. A bare ``import vllm`` succeeds even when ``vllm._C`` (or a
sibling like ``vllm._C_stable_libtorch``) is ABI-broken, so
``disable_broken_vllm`` never fired and the failure surfaced uncaught later
inside ``fast_inference_setup``.

Runs under the GPU-free ``tests/conftest.py`` with a synthetic vLLM."""

from __future__ import annotations

import contextlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types

import pytest


_LIBCUDART_ERROR = "libcudart.so.13: cannot open shared object file: No such file or directory"


class _ExtensionLoader(importlib.abc.Loader):
    """A compiled extension that either loads cleanly or fails on dlopen."""

    def __init__(self, broken, error):
        self.broken = broken
        self.error = error

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if self.broken:
            raise ImportError(self.error)


class _FakeVllmFinder(importlib.abc.MetaPathFinder):
    """A lazy-loading vLLM: ``import vllm`` is fine, and each ``vllm._*``
    extension is present-and-healthy, present-but-ABI-broken, or absent —
    mirroring how real vLLM only loads ``_C`` & friends when used."""

    def __init__(self, present, broken, error):
        self.present = present  # extensions this build ships
        self.broken = broken  # subset of present that fails to dlopen
        self.error = error  # dlopen message the broken ones raise

    def find_spec(
        self,
        fullname,
        path = None,
        target = None,
    ):
        if fullname in self.present:
            return importlib.machinery.ModuleSpec(
                name = fullname,
                loader = _ExtensionLoader(broken = fullname in self.broken, error = self.error),
                is_package = False,
            )
        return None  # absent -> ModuleNotFoundError, which the guard ignores


@contextlib.contextmanager
def _fake_vllm(
    present,
    broken,
    error = _LIBCUDART_ERROR,
):
    """Install a synthetic lazy vLLM and restore every global the guard
    touches (VLLM_BROKEN, the find_spec patch + meta-path blocker, and the
    vllm* sys.modules entries) on exit."""
    from unsloth import import_fixes

    submodules = import_fixes._VLLM_COMPILED_EXTENSIONS
    saved_meta_path = list(sys.meta_path)
    saved_find_spec = importlib.util.find_spec
    saved_broken = import_fixes.VLLM_BROKEN
    saved_modules = {n: sys.modules.get(n) for n in ("vllm", *submodules)}
    try:
        import_fixes.VLLM_BROKEN = False
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.__path__ = []
        fake_vllm.__spec__ = importlib.machinery.ModuleSpec("vllm", loader = None, is_package = True)
        sys.modules["vllm"] = fake_vllm
        for name in submodules:
            sys.modules.pop(name, None)
        sys.meta_path.insert(0, _FakeVllmFinder(present, broken, error))
        yield import_fixes
    finally:
        import_fixes.VLLM_BROKEN = saved_broken
        sys.meta_path[:] = saved_meta_path
        importlib.util.find_spec = saved_find_spec
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.mark.parametrize(
    "broken_ext",
    ["vllm._C", "vllm._C_stable_libtorch"],
    ids = ["core_C", "sibling_C_stable_libtorch"],
)
def test_disable_broken_vllm_detects_lazy_loaded_broken_extension(broken_ext):
    # A CUDA-major mismatch breaks every extension, but the break can surface
    # through any one vLLM lazily loads — _C or a sibling. Either must be caught.
    present = {"vllm._C", "vllm._C_stable_libtorch"}
    with _fake_vllm(present = present, broken = {broken_ext}) as import_fixes:
        detected = import_fixes.disable_broken_vllm()

        assert detected is True, (
            f"disable_broken_vllm missed an ABI-broken {broken_ext} behind a "
            "lazily-importable vllm package — issue #6590 would resurface."
        )
        assert import_fixes.VLLM_BROKEN is True
        # Once disabled, vLLM must look absent so callers fall back cleanly.
        assert importlib.util.find_spec("vllm") is None


@pytest.mark.parametrize(
    "error",
    [
        "libnccl.so.2: cannot open shared object file: No such file or directory",
        "libcuda.so.1: cannot open shared object file: No such file or directory",
    ],
    ids = ["libnccl", "libcuda"],
)
def test_disable_broken_vllm_detects_non_cudart_so_failure(error):
    # Force-loading vllm._C directly raises the bare loader error, without
    # vLLM's "vllm._C" wrapper. A CUDA-major mismatch can surface through a
    # linked .so other than libcudart (libnccl, libcuda, ...), which the old
    # libcudart/libcublas/libnvrtc allow-list would have let slip through.
    with _fake_vllm(present = {"vllm._C"}, broken = {"vllm._C"}, error = error) as import_fixes:
        detected = import_fixes.disable_broken_vllm()

        assert detected is True, (
            f"disable_broken_vllm missed a present-but-broken vllm._C raising "
            f"{error!r} — vLLM would be left enabled and crash later."
        )
        assert import_fixes.VLLM_BROKEN is True


@pytest.mark.parametrize(
    "present",
    [{"vllm._C"}, {"vllm._C", "vllm._C_stable_libtorch", "vllm._moe_C"}],
    ids = ["core_only", "all_present"],
)
def test_disable_broken_vllm_keeps_healthy_vllm_enabled(present):
    # A normal install: every extension that's present loads cleanly. Whether
    # only _C is built or _C plus siblings are, the probe must not disable it —
    # neither a ModuleNotFoundError on an absent sibling nor a clean load of an
    # extra present one may be mistaken for an ABI break.
    with _fake_vllm(present = present, broken = set()) as import_fixes:
        detected = import_fixes.disable_broken_vllm()

        assert detected is False
        assert import_fixes.VLLM_BROKEN is False
        assert importlib.util.find_spec("vllm") is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

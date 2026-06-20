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

"""Behavioural tests for ``fix_vllm_lora_warmup_rank`` (unsloth/import_fixes.py).

vLLM's v1 ``LoRAModelRunnerMixin.maybe_setup_dummy_loras()`` calls
``self.lora_manager.get_dummy_lora_warmup_rank(default_rank)``; older
``unsloth_zoo`` LoRA-manager ports don't implement it, so ``fast_inference``
LoRA warmup crashes with ``AttributeError`` (issue #6114). The shim wraps
``maybe_setup_dummy_loras`` to backfill the identity default on the *live*
manager class.

These reproduce the crash and the fix against a synthetic mixin/manager that
mirrors vLLM's exact call shape, so they exercise the shim's logic on the
GPU-free harness without vLLM (or ``unsloth_zoo``) installed."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import pathlib
import sys
import types
from contextlib import contextmanager

import pytest


def _load_import_fixes():
    """Load ``unsloth/import_fixes.py`` standalone.

    Importing ``unsloth.import_fixes`` the normal way executes the heavy
    ``unsloth`` package ``__init__`` (which needs unsloth_zoo / a real
    accelerator). The module itself is dependency-light and patches vLLM
    lazily, so load it directly from source for a GPU-free unit test."""
    path = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "import_fixes.py"
    spec = importlib.util.spec_from_file_location(
        "unsloth_import_fixes_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IMPORT_FIXES = _load_import_fixes()


def _make_classes():
    """Fresh (Mixin, Manager, Cfg) classes per call so a backfilled method on
    one test's manager class can't leak into another test."""

    class PortManager:
        """Mimics an older unsloth_zoo port: no ``get_dummy_lora_warmup_rank``."""

        def __init__(self):
            self.added_ranks = []
            self.removed = False

        @contextmanager
        def dummy_lora_cache(self):
            yield

        def add_dummy_lora(self, lora_request, rank):
            self.added_ranks.append(rank)

        def remove_all_adapters(self):
            self.removed = True

    class Cfg:
        def __init__(self, max_loras=1, max_lora_rank=16):
            self.max_loras = max_loras
            self.max_lora_rank = max_lora_rank

    class LoRAModelRunnerMixin:
        """Stand-in for vLLM's mixin with the real ``maybe_setup_dummy_loras``
        call shape (the ``get_dummy_lora_warmup_rank`` hop is what breaks)."""

        def __init__(self):
            self.lora_manager = PortManager()

        @contextmanager
        def maybe_setup_dummy_loras(self, lora_config, remove_lora=True):
            if lora_config is None:
                yield
                return
            assert self.lora_manager is not None, "LoRA is not enabled"
            warmup_rank = (
                lora_config.max_lora_rank if lora_config.max_lora_rank < 8 else 8
            )
            # The call that AttributeErrors on older unsloth_zoo ports (#6114):
            warmup_rank = self.lora_manager.get_dummy_lora_warmup_rank(warmup_rank)
            with self.lora_manager.dummy_lora_cache():
                for lora_id in range(1, lora_config.max_loras + 1):
                    self.lora_manager.add_dummy_lora(lora_id, rank=warmup_rank)
                yield
            if remove_lora:
                self.lora_manager.remove_all_adapters()

    return LoRAModelRunnerMixin, PortManager, Cfg


def _fake_module(mixin_cls):
    module = types.ModuleType("unsloth_test_fake_vllm_lora_mixin")
    module.LoRAModelRunnerMixin = mixin_cls
    return module


def test_unpatched_mixin_reproduces_issue_6114():
    Mixin, _Manager, Cfg = _make_classes()
    inst = Mixin()
    with pytest.raises(AttributeError, match="get_dummy_lora_warmup_rank"):
        with inst.maybe_setup_dummy_loras(Cfg(max_loras=2, max_lora_rank=16)):
            pass


def test_patch_backfills_identity_default_on_live_manager():
    Mixin, Manager, Cfg = _make_classes()
    IMPORT_FIXES._unsloth_patch_lora_model_runner_mixin(_fake_module(Mixin))

    inst = Mixin()
    with inst.maybe_setup_dummy_loras(Cfg(max_loras=2, max_lora_rank=16)):
        pass

    # min(max_lora_rank=16, 8) == 8, returned unchanged by the identity hook.
    assert inst.lora_manager.added_ranks == [8, 8]
    assert inst.lora_manager.removed is True
    # The hook now lives on the manager class and is a pass-through.
    assert hasattr(Manager, IMPORT_FIXES._VLLM_LORA_WARMUP_RANK_METHOD)
    assert Manager().get_dummy_lora_warmup_rank(5) == 5


def test_patch_preserves_a_manager_that_already_has_the_hook():
    Mixin, Manager, Cfg = _make_classes()

    def lowered(self, default_rank):
        return min(default_rank, 4)

    Manager.get_dummy_lora_warmup_rank = lowered
    IMPORT_FIXES._unsloth_patch_lora_model_runner_mixin(_fake_module(Mixin))

    inst = Mixin()
    with inst.maybe_setup_dummy_loras(Cfg(max_loras=1, max_lora_rank=16)):
        pass

    # The shim must not clobber a manager that implements its own hook.
    assert inst.lora_manager.added_ranks == [4]
    assert Manager.get_dummy_lora_warmup_rank is lowered


def test_patch_is_idempotent():
    Mixin, _Manager, _Cfg = _make_classes()
    module = _fake_module(Mixin)

    IMPORT_FIXES._unsloth_patch_lora_model_runner_mixin(module)
    first = Mixin.__dict__["maybe_setup_dummy_loras"]
    IMPORT_FIXES._unsloth_patch_lora_model_runner_mixin(module)
    second = Mixin.__dict__["maybe_setup_dummy_loras"]

    assert first is second  # not double-wrapped
    assert getattr(first, IMPORT_FIXES._VLLM_LORA_WARMUP_SHIM_SENTINEL, False)


def test_lora_disabled_path_is_untouched():
    Mixin, _Manager, _Cfg = _make_classes()
    IMPORT_FIXES._unsloth_patch_lora_model_runner_mixin(_fake_module(Mixin))

    inst = Mixin()
    with inst.maybe_setup_dummy_loras(None):  # lora_config is None
        pass
    assert inst.lora_manager.added_ranks == []


def _count_sentinel_finders():
    return sum(
        1
        for f in sys.meta_path
        if getattr(f, IMPORT_FIXES._VLLM_LORA_WARMUP_SHIM_SENTINEL, False)
    )


@contextmanager
def _faked_vllm(extra_modules=()):
    """Put a stand-in ``vllm`` (and optional submodules) in ``sys.modules`` so
    the shim's ``find_spec("vllm")`` guard passes, then restore on exit."""
    if importlib.util.find_spec("vllm") is not None:
        pytest.skip("vllm present: exercise against the real module elsewhere")
    names = ("vllm", *extra_modules)
    saved = {name: sys.modules.get(name) for name in names}
    meta_before = list(sys.meta_path)
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []  # mark as a package
    # find_spec("vllm") reads __spec__ for an already-imported module.
    fake_vllm.__spec__ = importlib.machinery.ModuleSpec(
        "vllm", loader=None, is_package=True
    )
    sys.modules["vllm"] = fake_vllm
    try:
        yield
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        sys.meta_path[:] = [
            f
            for f in sys.meta_path
            if f in meta_before
            or not getattr(f, IMPORT_FIXES._VLLM_LORA_WARMUP_SHIM_SENTINEL, False)
        ]


def test_fix_installs_one_post_import_finder_and_is_idempotent():
    with _faked_vllm():
        IMPORT_FIXES.fix_vllm_lora_warmup_rank()
        assert _count_sentinel_finders() == 1
        # Second call must not stack a second finder.
        IMPORT_FIXES.fix_vllm_lora_warmup_rank()
        assert _count_sentinel_finders() == 1


def test_fix_patches_in_place_when_mixin_already_imported():
    Mixin, _Manager, Cfg = _make_classes()
    mod_name = IMPORT_FIXES._VLLM_LORA_MIXIN_MODULE
    with _faked_vllm(extra_modules=(mod_name,)):
        # vLLM (and the mixin module) already imported before us -> patch in
        # place, no finder needed.
        sys.modules[mod_name] = _fake_module(Mixin)
        IMPORT_FIXES.fix_vllm_lora_warmup_rank()
        assert _count_sentinel_finders() == 0

        inst = Mixin()
        with inst.maybe_setup_dummy_loras(Cfg(max_loras=1, max_lora_rank=4)):
            pass
        assert inst.lora_manager.added_ranks == [4]  # min(4, 8) == 4

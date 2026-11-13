# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Make accelerate's ``init_on_device`` / ``init_empty_weights`` per-thread.

accelerate swaps ``torch.nn.Module.register_parameter`` process-wide for the length of the ``with`` block and restores
whatever it saved on exit. Studio builds meta skeletons from several threads at once (a load, and the
``download-plan`` fit checks the Images page fires while it runs), so (a) a module another thread builds meanwhile
gets meta parameters ("N tensor(s) ... were not in the file", "Cannot copy out of meta tensor"), and (b) two
overlapping blocks that exit in the wrong order leave the meta patch installed for good: every later load in the
process fails until a restart.

``install()`` puts one permanent dispatcher on ``register_parameter`` that moves a parameter to the meta (or other)
device only while the *calling thread* is inside ``init_on_device``, and replaces accelerate's ``init_on_device`` with
a version that keeps that state thread-local. Idempotent; a missing accelerate is a no-op."""

from __future__ import annotations

import contextlib
import importlib.abc
import importlib.util
import sys
import threading
from typing import Any, Optional

_tls = threading.local()
_install_lock = threading.Lock()
_installed = False


def _device_stack() -> list:
    stack = getattr(_tls, "devices", None)
    if stack is None:
        stack = _tls.devices = []
    return stack


def install() -> bool:
    """Patch once; True when the per-thread version is in place."""
    global _installed
    with _install_lock:
        if _installed:
            return True
        try:
            import torch.nn as nn
            from accelerate import big_modeling
        except Exception:  # noqa: BLE001 -- no accelerate, nothing to guard
            return False
        try:
            from accelerate.utils import parse_flag_from_env
        except Exception:  # noqa: BLE001
            parse_flag_from_env = None

        original_register_parameter = nn.Module.register_parameter

        def register_parameter(module: Any, name: str, param: Any) -> None:
            original_register_parameter(module, name, param)
            stack = getattr(_tls, "devices", None)
            if not stack or param is None:
                return
            # The body of accelerate's register_empty_parameter, for this thread only.
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            is_hf_initialized = kwargs.pop("_is_hf_initialized", None)
            module._parameters[name] = param_cls(module._parameters[name].to(stack[-1]), **kwargs)
            if is_hf_initialized is not None:
                module._parameters[name]._is_hf_initialized = is_hf_initialized

        @contextlib.contextmanager
        def init_on_device(device: Any, include_buffers: Optional[bool] = None):
            if include_buffers is None:
                include_buffers = (
                    parse_flag_from_env("ACCELERATE_INIT_INCLUDE_BUFFERS", False)
                    if parse_flag_from_env is not None
                    else False
                )
            if include_buffers:
                # accelerate's own path: a torch.device context, which is already thread-local.
                with device:
                    yield
                return
            stack = _device_stack()
            stack.append(device)
            try:
                yield
            finally:
                stack.pop()

        nn.Module.register_parameter = register_parameter
        big_modeling.init_on_device = init_on_device
        try:
            import accelerate
            if getattr(accelerate, "init_on_device", None) is not None:
                accelerate.init_on_device = init_on_device
        except Exception:  # noqa: BLE001
            pass
        _installed = True
        return True


class _PatchOnImport(importlib.abc.MetaPathFinder):
    """Runs ``install()`` right after ``accelerate.big_modeling`` executes, so startup never imports torch for this."""

    def find_spec(
        self,
        fullname,
        path,
        target = None,
    ):
        if fullname != "accelerate.big_modeling":
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        original_exec = loader.exec_module

        def exec_module(module):
            original_exec(module)
            sys.modules[fullname] = module
            with contextlib.suppress(ValueError):
                sys.meta_path.remove(self)
            install()

        loader.exec_module = exec_module
        return spec


def install_on_import() -> None:
    """``install()`` now when accelerate is already loaded, else as soon as it is imported."""
    if "accelerate.big_modeling" in sys.modules:
        install()
        return
    if not any(isinstance(f, _PatchOnImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchOnImport())

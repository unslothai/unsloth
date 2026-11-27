# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Make accelerate's ``init_on_device`` / ``init_empty_weights`` per-thread.

accelerate swaps ``nn.Module.register_parameter`` process-wide for the block and restores what it saved on exit, so
concurrent skeleton builds (a load plus Images ``download-plan`` fit checks) leak meta parameters into other threads
and, when blocks exit out of order, leave the patch installed for the life of the process.

Behaviour change: a thread started inside a block no longer inherits it (no Studio block starts a thread)."""

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
            from accelerate.utils import parse_flag_from_env
        except Exception:  # noqa: BLE001 -- no accelerate, nothing to guard
            return False

        original_register_parameter = nn.Module.register_parameter

        def register_parameter(module: Any, name: str, param: Any) -> None:
            original_register_parameter(module, name, param)
            stack = getattr(_tls, "devices", None)
            if not stack or stack[-1] is None or param is None:
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
                include_buffers = parse_flag_from_env("ACCELERATE_INIT_INCLUDE_BUFFERS", False)
            # None in buffers mode: the torch.device context places parameters; re-wrapping them breaks tied weights.
            stack = _device_stack()
            stack.append(None if include_buffers else device)
            try:
                if include_buffers:
                    with device:
                        yield
                else:
                    yield
            finally:
                stack.pop()

        nn.Module.register_parameter = register_parameter
        big_modeling.init_on_device = init_on_device
        # Mid-import of accelerate this isn't bound yet; its own `from .big_modeling import` picks ours up.
        accelerate = sys.modules.get("accelerate")
        if getattr(accelerate, "init_on_device", None) is not None:
            accelerate.init_on_device = init_on_device
        _installed = True
        return True


class _PatchOnImport(importlib.abc.MetaPathFinder):
    """Runs ``install()`` right after ``accelerate.big_modeling`` executes, so startup does not import torch."""

    def find_spec(
        self,
        fullname,
        path,
        target = None,
    ):
        if fullname != "accelerate.big_modeling" or _installed or getattr(_tls, "finding", False):
            return None
        # Stays on sys.meta_path (a no-op once installed): removing it races other threads' imports.
        _tls.finding = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            _tls.finding = False
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        original_exec = loader.exec_module

        def exec_module(module):
            original_exec(module)
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

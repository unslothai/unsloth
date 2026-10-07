# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

# Low overhead eager launches: the first call per specialization goes through Triton, later ones
# launch the compiled kernel directly, skipping JITFunction's per-call argument binding. The key
# covers what Triton specializes on: dtypes, 16 byte alignment, ints == 1 / % 16 / > int32, options.

import hashlib
import os

import torch
import triton

__all__ = [
    "launch",
    "relaunch",
    "tag_compile_cache",
]

_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
_ENABLED = (
    _raw_stream is not None
    and torch.version.hip is None
    and os.environ.get("UNSLOTH_TRITON_FAST_LAUNCH", "1") != "0"
)
_CACHE = {}
_knobs = getattr(triton, "knobs", None)


def _active(hook):
    # Triton 3.5+ always installs a HookChain, empty unless something registered a hook.
    return hook is not None and getattr(hook, "calls", True) != []


def _hooks_set():
    # Profilers (e.g. proton) hook Triton launches; those launches keep Triton's own path.
    if _knobs is not None:
        runtime = _knobs.runtime
        return _active(runtime.launch_enter_hook) or _active(runtime.launch_exit_hook)
    from triton.compiler import CompiledKernel
    return _active(getattr(CompiledKernel, "launch_enter_hook", None)) or _active(
        getattr(CompiledKernel, "launch_exit_hook", None)
    )


def relaunch(entry, grid, device_index, args):
    """Repeat a launch whose cache entry launch() returned, with args matching the original in
    dtypes, 16 byte alignment and integer specialization, on device_index (the current device).
    False when the caller must use launch() instead."""
    if not _ENABLED or _hooks_set():
        return False
    compiled, tail = entry
    compiled.run(
        grid[0],
        grid[1] if len(grid) > 1 else 1,
        grid[2] if len(grid) > 2 else 1,
        _raw_stream(device_index),
        compiled.function,
        compiled.packed_metadata,
        None,
        None,
        None,
        *args,
        *tail,
    )
    return True


def launch(kernel, grid, args, n_tensors, constexprs, device_index, **options):
    """kernel[grid](*args, **constexprs, **options) on the current stream of device_index, which
    must be the current device. args are the runtime arguments in signature order, before every
    constexpr: n_tensors tensors, then integers. Returns the cache entry relaunch() takes, or None."""
    global _ENABLED
    if _ENABLED and not _hooks_set():
        key = (
            kernel,
            device_index,
            tuple(constexprs.values()),
            tuple(options.values()),
            tuple([(arg.dtype, arg.data_ptr() & 15) for arg in args[:n_tensors]]),
            tuple([(arg == 1, arg & 15, arg < (1 << 31)) for arg in args[n_tensors:]]),
        )
        entry = _CACHE.get(key)
        if entry is not None:
            compiled, tail = entry
            try:
                compiled.run(
                    grid[0],
                    grid[1] if len(grid) > 1 else 1,
                    grid[2] if len(grid) > 2 else 1,
                    _raw_stream(device_index),
                    compiled.function,
                    compiled.packed_metadata,
                    None,
                    None,
                    None,
                    *args,
                    *tail,
                )
                return entry
            except Exception:
                # A Triton build whose launcher takes other arguments: use its own path from now.
                _ENABLED = False
                _CACHE.clear()
        compiled = kernel[grid](*args, **constexprs, **options)
        if _ENABLED and hasattr(compiled, "packed_metadata"):
            names = kernel.arg_names[len(args) :]
            if len(names) == len(constexprs):
                compiled.run  # loads the module now, so the fast path never does
                entry = _CACHE[key] = (compiled, tuple(constexprs[name] for name in names))
                return entry
        return None
    kernel[grid](*args, **constexprs, **options)
    return None


def tag_compile_cache(path):
    """Add a hash of a kernel file to torch.compile's cache key. Inductor's FX graph cache keys a
    torch.library.triton_op call without the Triton source behind it, so after an Unsloth upgrade
    a warm cache would keep serving the previous kernel."""
    config = getattr(torch.compiler, "config", None)
    if config is None or not hasattr(config, "cache_key_tag"):
        return
    with open(path, "rb") as file:
        tag = f"unsloth/{os.path.basename(path)}:{hashlib.sha256(file.read()).hexdigest()[:16]}"
    tags = [t for t in config.cache_key_tag.split(",") if t]
    if tag not in tags:
        config.cache_key_tag = ",".join(tags + [tag])

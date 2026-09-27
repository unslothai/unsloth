# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

# Low overhead eager launches for the NF4 kernels. Triton's JITFunction launch re-binds the
# arguments and rebuilds the specialization key on every call, about 12 us of CPU against about
# 5 us for launching the compiled kernel directly. A 1B QLoRA step makes around a thousand 4bit
# launches, so on a fast GPU the difference shows up in the step time.
#
# The first call for a specialization goes through Triton, which compiles and launches the kernel
# and returns the compiled kernel; later calls with the same specialization launch it directly.
# The key covers everything Triton specializes on here: tensor dtypes and 16 byte alignment,
# integers equal to 1, divisible by 16 or outside int32, the constexprs and the launch options.

import os

import torch
import triton

__all__ = [
    "launch",
]

_ENABLED = torch.version.hip is None and os.environ.get("UNSLOTH_TRITON_FAST_LAUNCH", "1") != "0"
_CACHE = {}
_raw_stream = torch._C._cuda_getCurrentRawStream
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


def launch(kernel, grid, args, n_tensors, constexprs, device_index, **options):
    """kernel[grid](*args, **constexprs, **options) on the current stream of device_index, which
    must be the current device. args are the runtime arguments in signature order, before every
    constexpr: n_tensors tensors, then integers."""
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
                return
            except Exception:
                # A Triton build whose launcher takes other arguments: use its own path from now.
                _ENABLED = False
                _CACHE.clear()
        compiled = kernel[grid](*args, **constexprs, **options)
        if _ENABLED and hasattr(compiled, "packed_metadata"):
            names = kernel.arg_names[len(args) :]
            if len(names) == len(constexprs):
                compiled.run  # loads the module now, so the fast path never does
                _CACHE[key] = (compiled, tuple(constexprs[name] for name in names))
        return
    kernel[grid](*args, **constexprs, **options)

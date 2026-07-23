# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Standalone free-VRAM probe for the bundled ggml Vulkan backend.

Run in a short-lived subprocess (``python _vulkan_probe.py <bindir>``) so the
Vulkan instance never lives in the long-running backend process. Loads the
bundled ggml Vulkan backend from ``<bindir>`` and prints one
``<idx>\\t<free_bytes>\\t<is_igpu>\\t<total_bytes>\\t<name>`` line per device to
stdout. Indices are ggml's own Vulkan device ordinals, which need not match
nvidia-smi order. ``is_igpu`` (from ggml's device type) is ``1`` for an
integrated GPU sharing system RAM. ``total_bytes`` is the device-local heap;
the reader uses it to reserve absolute headroom on a discrete card (parity
with the CUDA/ROCm fit) and ignores it for an iGPU, whose "VRAM" is shared
system RAM. ``name`` is ggml's device description (the marketing name, e.g.
"AMD Radeon RX 9070 XT"); empty when the registry lookup fails.

Uses only the standard library so it stays runnable as a bare script.
"""

import ctypes
import os
import sys

# ggml_backend_dev_type enum (ggml-backend.h): CPU=0, GPU=1, IGPU=2, ...
_GGML_BACKEND_DEVICE_TYPE_IGPU = 2


def _igpu_flags_and_names(base, lib, count: int) -> tuple[list[bool], list[str]]:
    """Per-device integrated-GPU flags and descriptions via ggml's backend registry.

    The Vulkan reg enumerates devices in the same order as
    ``ggml_backend_vk_get_device_memory`` (each context uses ``ctx->device =
    i``), so reg index == device ordinal. Returns all-False / empty-name on any
    failure so the reader never over-caps a discrete card and the memory
    readings still get through.
    """
    flags = [False] * count
    names = [""] * count
    try:
        lib.ggml_backend_vk_reg.restype = ctypes.c_void_p
        lib.ggml_backend_vk_reg.argtypes = []
        base.ggml_backend_reg_dev_count.restype = ctypes.c_size_t
        base.ggml_backend_reg_dev_count.argtypes = [ctypes.c_void_p]
        base.ggml_backend_reg_dev_get.restype = ctypes.c_void_p
        base.ggml_backend_reg_dev_get.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        base.ggml_backend_dev_type.restype = ctypes.c_int
        base.ggml_backend_dev_type.argtypes = [ctypes.c_void_p]
        base.ggml_backend_dev_description.restype = ctypes.c_char_p
        base.ggml_backend_dev_description.argtypes = [ctypes.c_void_p]

        reg = lib.ggml_backend_vk_reg()
        if not reg:
            return flags, names
        dev_count = base.ggml_backend_reg_dev_count(reg)
        for i in range(min(count, dev_count)):
            dev = base.ggml_backend_reg_dev_get(reg, i)
            if dev:
                flags[i] = base.ggml_backend_dev_type(dev) == _GGML_BACKEND_DEVICE_TYPE_IGPU
                desc = base.ggml_backend_dev_description(dev)
                if desc:
                    # Tabs/newlines would corrupt the line protocol; spaces are safe.
                    names[i] = desc.decode("utf-8", errors = "replace").replace("\t", " ").replace("\n", " ").strip()
    except Exception:
        # Best-effort: any failure degrades to "discrete"/"unnamed" so the
        # memory readings still get through instead of crashing the probe.
        pass
    return flags, names


def main() -> int:
    if len(sys.argv) < 2:
        return 0
    bindir = sys.argv[1]

    # Hold add_dll_directory's handle for the rest of main() (the documented
    # idiom) so bindir stays on the search path while the sibling ggml DLLs
    # resolve below.
    _dll_dir = None
    if sys.platform == "win32":
        base_name, vk_name = "ggml-base.dll", "ggml-vulkan.dll"
        try:
            _dll_dir = os.add_dll_directory(bindir)
        except Exception:
            pass
    else:
        base_name, vk_name = "libggml-base.so", "libggml-vulkan.so"

    # RTLD_GLOBAL exposes ggml-base's symbols to ggml-vulkan on POSIX. getattr
    # falls back to 0 where the flag doesn't exist (Windows CDLL ignores mode).
    _rtld_global = getattr(ctypes, "RTLD_GLOBAL", 0)
    try:
        base = ctypes.CDLL(os.path.join(bindir, base_name), mode = _rtld_global)
        lib = ctypes.CDLL(os.path.join(bindir, vk_name), mode = _rtld_global)
    except OSError as e:
        print(f"ggml-vulkan load failed: {e}", file = sys.stderr)
        return 1

    lib.ggml_backend_vk_get_device_count.restype = ctypes.c_int
    lib.ggml_backend_vk_get_device_count.argtypes = []
    lib.ggml_backend_vk_get_device_memory.restype = None
    lib.ggml_backend_vk_get_device_memory.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]

    count = lib.ggml_backend_vk_get_device_count()
    igpu, names = _igpu_flags_and_names(base, lib, count)
    rows = []
    for i in range(count):
        free, total = ctypes.c_size_t(0), ctypes.c_size_t(0)
        lib.ggml_backend_vk_get_device_memory(i, ctypes.byref(free), ctypes.byref(total))
        rows.append("%d\t%d\t%d\t%d\t%s" % (i, free.value, int(igpu[i]), total.value, names[i]))
    sys.stdout.write("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

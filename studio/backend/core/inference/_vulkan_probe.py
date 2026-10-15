# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Standalone free-VRAM probe for the bundled ggml Vulkan backend.

Run in a short-lived subprocess (``python _vulkan_probe.py <bindir>``) so the
Vulkan instance never lives in the long-running backend process. Loads the
bundled ggml Vulkan backend from ``<bindir>`` and prints one
``<idx>\\t<free_bytes>\\t<is_igpu>\\t<total_bytes>\\t<name>\\t<type_known>`` line per device to
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

# ggml_backend_dev_type enum (ggml-backend.h): CPU=0, GPU=1, IGPU=2.
_GGML_BACKEND_DEVICE_TYPE_IGPU = 2
# winbase.h
_SEM_FAILCRITICALERRORS = 0x0001


def _igpu_flags_and_names(base, lib, count: int) -> tuple[list[bool], list[str], list[bool]]:
    """Third list marks flags actually read: a DirectIO choice on an unread type buffers iGPU weights."""
    flags = [False] * count
    names = [""] * count
    known = [False] * count
    try:
        lib.ggml_backend_vk_reg.restype = ctypes.c_void_p
        lib.ggml_backend_vk_reg.argtypes = []
        base.ggml_backend_reg_dev_count.restype = ctypes.c_size_t
        base.ggml_backend_reg_dev_count.argtypes = [ctypes.c_void_p]
        base.ggml_backend_reg_dev_get.restype = ctypes.c_void_p
        base.ggml_backend_reg_dev_get.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        base.ggml_backend_dev_type.restype = ctypes.c_int
        base.ggml_backend_dev_type.argtypes = [ctypes.c_void_p]
        reg = lib.ggml_backend_vk_reg()
        if not reg:
            return flags, names, known
        dev_count = base.ggml_backend_reg_dev_count(reg)
    except Exception:
        return flags, names, known

    # Outside the try above: a ggml-base without the description symbol must still read iGPU flags.
    name_functions = []
    for symbol in ("ggml_backend_dev_description", "ggml_backend_dev_name"):
        try:
            function = getattr(base, symbol)
            function.restype = ctypes.c_char_p
            function.argtypes = [ctypes.c_void_p]
            name_functions.append(function)
        except Exception:
            pass

    for i in range(min(count, dev_count)):
        try:
            dev = base.ggml_backend_reg_dev_get(reg, i)
        except Exception:
            continue
        if not dev:
            continue
        try:
            flags[i] = base.ggml_backend_dev_type(dev) == _GGML_BACKEND_DEVICE_TYPE_IGPU
            known[i] = True
        except Exception:
            pass
        for function in name_functions:
            try:
                raw_name = function(dev)
            except Exception:
                continue
            if raw_name:
                name = raw_name.decode("utf-8", errors = "replace")
                names[i] = name.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()
                break
    return flags, names, known


def main() -> int:
    if len(sys.argv) < 2:
        return 0
    bindir = sys.argv[1]

    # Localized device names would raise under cp1252 stdout and lose the inventory.
    try:
        sys.stdout.reconfigure(encoding = "utf-8", errors = "replace")
    except Exception:
        pass

    # Hold add_dll_directory's handle so bindir stays on the search path.
    _dll_dir = None
    if sys.platform == "win32":
        base_name, vk_name = "ggml-base.dll", "ggml-vulkan.dll"
        try:
            _dll_dir = os.add_dll_directory(bindir)
        except Exception:
            pass
        # Else a pre-1.1 vulkan-1.dll pops a modal "Entry point not found" box.
        try:
            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.SetErrorMode(kernel32.GetErrorMode() | _SEM_FAILCRITICALERRORS)
        except Exception:
            pass
    else:
        base_name, vk_name = "libggml-base.so", "libggml-vulkan.so"

    def _find_lib(directory, stem):
        """Find an unversioned library or a versioned runtime soname."""
        path = os.path.join(directory, stem)
        if os.path.isfile(path):
            return path
        for entry in sorted(os.listdir(directory)):
            if entry.startswith(stem + "."):
                return os.path.join(directory, entry)
        return None

    # RTLD_GLOBAL exposes ggml-base's symbols to ggml-vulkan on POSIX.
    _rtld_global = getattr(ctypes, "RTLD_GLOBAL", 0)
    base_path = _find_lib(bindir, base_name)
    vk_path = _find_lib(bindir, vk_name)
    if not base_path or not vk_path:
        print(f"ggml-vulkan load failed: library not found in {bindir}", file = sys.stderr)
        return 1
    try:
        base = ctypes.CDLL(base_path, mode = _rtld_global)
        lib = ctypes.CDLL(vk_path, mode = _rtld_global)
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
    igpu, names, known = _igpu_flags_and_names(base, lib, count)
    rows = []
    for i in range(count):
        free, total = ctypes.c_size_t(0), ctypes.c_size_t(0)
        lib.ggml_backend_vk_get_device_memory(i, ctypes.byref(free), ctypes.byref(total))
        rows.append(
            "%d\t%d\t%d\t%d\t%s\t%d"
            % (i, free.value, int(igpu[i]), total.value, names[i], int(known[i]))
        )
    sys.stdout.write("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

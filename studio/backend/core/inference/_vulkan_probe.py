"""Standalone free-VRAM probe for the bundled ggml Vulkan backend.

Run in a short-lived subprocess (``python _vulkan_probe.py <bindir>``) so the
Vulkan instance never lives in the long-running backend process. Loads the
bundled ggml Vulkan backend from ``<bindir>`` and prints one
``<idx>\\t<free_bytes>\\t<total_bytes>`` line per device to stdout. The indices
are ggml's own Vulkan device ordinals (the space GGML_VK_VISIBLE_DEVICES
expects), which need not match nvidia-smi order.

Uses only the standard library so it stays runnable as a bare script without
importing the backend package.
"""
import ctypes
import os
import sys


def main() -> int:
    if len(sys.argv) < 2:
        return 0
    bindir = sys.argv[1]

    if sys.platform == "win32":
        base_name, vk_name = "ggml-base.dll", "ggml-vulkan.dll"
        try:
            os.add_dll_directory(bindir)
        except Exception:
            pass
    else:
        base_name, vk_name = "libggml-base.so", "libggml-vulkan.so"

    try:
        ctypes.CDLL(os.path.join(bindir, base_name), mode=ctypes.RTLD_GLOBAL)
        lib = ctypes.CDLL(os.path.join(bindir, vk_name), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return 0

    lib.ggml_backend_vk_get_device_count.restype = ctypes.c_int
    lib.ggml_backend_vk_get_device_count.argtypes = []
    lib.ggml_backend_vk_get_device_memory.restype = None
    lib.ggml_backend_vk_get_device_memory.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]

    rows = []
    for i in range(lib.ggml_backend_vk_get_device_count()):
        free, total = ctypes.c_size_t(0), ctypes.c_size_t(0)
        lib.ggml_backend_vk_get_device_memory(i, ctypes.byref(free), ctypes.byref(total))
        rows.append("%d\t%d\t%d" % (i, free.value, total.value))
    sys.stdout.write("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

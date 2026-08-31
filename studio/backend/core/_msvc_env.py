# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Gate torch.compile on Triton's C toolchain, on Windows (#7595). `import triton` succeeding does not
mean a compile will: Triton's AMD/HIP driver JIT-compiles `hip_utils.c` with `clang-cl`, which without
CRT headers dies on `'stdlib.h'` mid-run. It passes its own `/I` dirs, never `INCLUDE`: nothing to repair."""

from __future__ import annotations

import os
import sys
import logging

logger = logging.getLogger(__name__)


def _have_crt_headers() -> bool:
    """Only the compiler's own fallback; Triton's `/I` dirs are authoritative, see the module note."""
    for d in os.environ.get("INCLUDE", "").split(os.pathsep):
        if d and os.path.isfile(os.path.join(d, "stdlib.h")):
            return True
    return False


def _triton_finds_crt_headers() -> bool:
    """The dirs Triton will pass as `/I`, which is what the compile actually searches."""
    try:
        from triton.windows_utils import find_msvc_winsdk  # noqa: PLC0415
        _, inc_dirs, _ = find_msvc_winsdk()
    except Exception:  # noqa: BLE001 -- absent, older or broken Triton: fall back to INCLUDE
        logger.debug("Triton's MSVC/WinSDK discovery is unavailable", exc_info = True)
        return False
    return any(d and os.path.isfile(os.path.join(d, "stdlib.h")) for d in inc_dirs)


def _triton_is_triton_windows() -> bool:
    """XPU Triton and triton-windows own the same top-level name; only the distribution says which."""
    try:
        import importlib.metadata as md  # noqa: PLC0415
        dists = md.packages_distributions().get("triton") or ()
    except Exception:  # noqa: BLE001
        logger.debug("Could not resolve which distribution owns `triton`", exc_info = True)
        return False
    return any(d.lower().replace("_", "-") == "triton-windows" for d in dists)


def _rocm_clang_cl_present() -> bool:
    """`get_cc()` prefers this clang-cl over everything else once the ROCm wheel is installed."""
    import sysconfig  # noqa: PLC0415
    return os.path.isfile(
        os.path.join(
            sysconfig.get_path("platlib"), "_rocm_sdk_core", "lib", "llvm", "bin", "clang-cl.exe"
        )
    )


def _needs_msvc_headers() -> bool:
    """Only cl and clang-cl need them; `get_cc()` otherwise picks bundled TinyCC, which carries its own. The
    fallback needs both halves, since a ROCm-to-XPU repair strands the compiler without the Triton that
    selects it; unanswerable means ungated, because gating wrongly breaks a working box."""
    try:
        from triton.runtime.build import get_cc, is_clang_cl, is_msvc  # noqa: PLC0415
        cc = get_cc()
    except Exception:  # noqa: BLE001
        logger.debug("Triton's compiler selection is unavailable", exc_info = True)
        return _triton_is_triton_windows() and _rocm_clang_cl_present()
    return bool(is_msvc(cc) or is_clang_cl(cc))


def _toolchain_summary() -> str:
    """0 include dirs means no Visual Studio; several with no `stdlib.h` means a partial SDK."""
    try:
        from triton.runtime.build import get_cc  # noqa: PLC0415
        cc = os.path.basename(get_cc())
    except Exception:  # noqa: BLE001
        cc = "unknown"
    try:
        from triton.windows_utils import find_msvc_winsdk  # noqa: PLC0415
        _, inc_dirs, _ = find_msvc_winsdk()
    except Exception:  # noqa: BLE001
        inc_dirs = []
    logger.debug("Triton include dirs: %s", list(inc_dirs))
    return (
        f"compiler={cc}, Triton include dirs={len(inc_dirs)}, "
        f"INCLUDE={'set' if os.environ.get('INCLUDE') else 'unset'}"
    )


def crt_headers_reachable() -> bool:
    """Whether a Triton compile in THIS process will find the CRT headers. True off win32."""
    if sys.platform != "win32":
        return True
    if not _needs_msvc_headers():
        return True
    return _triton_finds_crt_headers() or _have_crt_headers()


def gate_torch_compile_on_windows(log: logging.Logger) -> None:
    """Workers call this before importing torch."""
    if sys.platform != "win32":
        return
    try:
        import triton  # noqa: F401, PLC0415
    except ImportError:
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        log.warning(
            "Triton not found on Windows — torch.compile disabled. "
            'Install for better performance: pip install "triton-windows<3.7"'
        )
        return

    try:
        reachable = crt_headers_reachable()
    except Exception:  # noqa: BLE001 -- a probe must not take down the worker it exists to protect
        logger.debug("The toolchain probe raised; leaving torch.compile alone", exc_info = True)
        reachable = True
    if reachable:
        log.info("Triton available — torch.compile enabled")
        return

    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    # This only turns off the compiles we own; Unsloth's @triton.jit kernels still need the toolchain.
    log.warning(
        "Triton is installed but its C toolchain has no CRT headers, so its "
        "clang-cl JIT would fail on 'stdlib.h' (#7595). torch.compile disabled; "
        "directly launched Triton kernels still need the toolchain. "
        "Install Visual Studio Build Tools (C++ workload): winget "
        'install Microsoft.VisualStudio.2022.BuildTools --override "--add '
        'Microsoft.VisualStudio.Workload.VCTools --includeRecommended". '
        "Probe saw: %s.",
        _toolchain_summary(),
    )
